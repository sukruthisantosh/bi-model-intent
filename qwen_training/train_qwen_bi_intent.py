#!/usr/bin/env python3
"""
Qwen SLM Training for BI Intent Discovery
=========================================

This script trains a Qwen model on Business Intelligence intent discovery using the prompt-based approach.

Model: Qwen2.5-0.5B (small, fast training)
Task: BI Intent Discovery (Planning + Discovery phases)
Dataset: 100 examples of BI questions with structured outputs
Training: Supervised Fine-tuning (SFT) with LoRA

Expected Output Format:
{
  "intent": "intents_discovery",
  "discovery_results": [
    {
      "step_id": "step_1",
      "sub_question": "...",
      "measures": [...],
      "dimensions": [...],
      "timegrain": null,
      "timeframe": null,
      "pattern": null,
      "segments": [],
      "breakdowns": [],
      "unmatched_intents": []
    }
  ]
}
"""

import json
import os
import gc
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import wandb
from tqdm.auto import tqdm
import numpy as np

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================

def install_requirements():
    """Install required packages."""
    import subprocess
    import sys
    
    packages = [
        "transformers>=4.40.0",
        "datasets>=2.16.0", 
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "wandb",
        "tqdm"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("✅ All packages installed successfully")

def check_gpu():
    """Check GPU availability and print info."""
    print(f"🚀 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - training will be slow!")

# ============================================================================
# CELL 2: Load Training Data
# ============================================================================

def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data)} training examples")
    
    # Show sample
    print("\n📝 Sample training example:")
    print(json.dumps(data[0], indent=2))
    
    return data

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8):
    """Split data into train/validation sets."""
    train_size = int(train_ratio * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"\n📈 Train examples: {len(train_data)}")
    print(f"📊 Validation examples: {len(val_data)}")
    
    return train_data, val_data

# ============================================================================
# CELL 3: Load Prompt Template
# ============================================================================

def load_prompt_template() -> str:
    """Load the prompt template for BI intent discovery."""
    prompt_template = """# Planning and Discovery Agent
You are an AI assistant specialized in analyzing natural language questions about business intelligence data and breaking them down into structured steps for query building.

## Your Role
You have two main phases of operation:
### Phase 1: Planning (for complex questions)
- Analyze the user's question to determine if it requires multi-step processing
- For complex questions, break them down into structured steps that can be used to build CTEs (Common Table Expressions)
- Identify dependencies between steps
- For simple questions, skip this phase and go directly to discovery

### Phase 2: Discovery
- Analyze the question (or planning steps) to identify BI concepts
- Map natural language terms to specific dimensions, measures, and filters
- Handle ambiguity by requesting clarification when needed

## Question Complexity Assessment
A question is COMPLEX if it contains ANY of these logical patterns:
1. **Implicit Dependencies**: When one concept depends on another
2. **Sequential Logic**: When steps must be performed in order
3. **Ranking/Selection Logic**: When filtering requires prior analysis
4. **Multi-Step Filtering**: When filters depend on other filters
5. **Comparative Analysis**: When comparing requires separate data gathering
6. **Time-Based Dependencies**: When time periods affect other queries

## Response Format
Respond with a JSON object containing your intent and the appropriate data structure based on the phase you're executing.

## Current Context
- User Question: {question}

## Instructions
1. Assess Question Complexity: Determine if this is a simple or complex question
2. For Complex Questions: Execute planning phase first, then discovery phase on the planning steps
3. For Simple Questions: Skip planning phase, execute discovery phase directly on the question
4. Handle Ambiguity: Request human input when terms are unclear
5. Use Available Tools: Use the appropriate tool based on your phase and intent

## Response Format
Respond with a JSON object containing your intent and the appropriate data structure based on the phase you're executing.

Output:"""
    
    return prompt_template

# ============================================================================
# CELL 4: Create Dataset Class
# ============================================================================

@dataclass
class TrainingExample:
    """Represents a single training example."""
    question: str
    expected_output: Dict[str, Any]
    
class BIIntentDataset(Dataset):
    """Dataset for BI Intent Discovery training."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Create the input prompt
        input_text = prompt_template.format(question=example['input'])
        
        # Create the expected output
        output_text = json.dumps(example['output'], ensure_ascii=False, separators=(',', ':'))
        
        # Combine input and output
        full_text = input_text + output_text
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

# ============================================================================
# CELL 5: Load Qwen Model and Tokenizer
# ============================================================================

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Load Qwen model and tokenizer with 4-bit quantization."""
    print(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"✅ Tokenizer loaded")
    print(f"📏 Vocabulary size: {tokenizer.vocab_size}")
    print(f"🔤 Special tokens: {tokenizer.special_tokens_map}")
    
    # Load model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    print(f"✅ Model loaded with 4-bit quantization")
    print(f"💾 Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer

# ============================================================================
# CELL 6: Setup LoRA for Efficient Fine-tuning
# ============================================================================

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning."""
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter
        lora_dropout=0.1,  # Dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print("✅ LoRA applied successfully")
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled")
    
    return model, lora_config

# ============================================================================
# CELL 7: Create Training Datasets
# ============================================================================

def create_datasets(train_data, val_data, tokenizer, max_length: int = 2048):
    """Create training and validation datasets."""
    train_dataset = BIIntentDataset(train_data, tokenizer, max_length)
    val_dataset = BIIntentDataset(val_data, tokenizer, max_length)
    
    print(f"✅ Training dataset created: {len(train_dataset)} examples")
    print(f"✅ Validation dataset created: {len(val_dataset)} examples")
    
    # Test a sample
    sample = train_dataset[0]
    print(f"\n📝 Sample input shape: {sample['input_ids'].shape}")
    print(f"📏 Sample length: {sample['input_ids'].shape[0]} tokens")
    
    # Decode sample to verify
    sample_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"\n📖 Sample decoded (first 500 chars):")
    print(sample_text[:500] + "...")
    
    return train_dataset, val_dataset

# ============================================================================
# CELL 8: Training Configuration
# ============================================================================

def create_training_args(output_dir: str = "./qwen-bi-intent-model", 
                        num_epochs: int = 3,
                        batch_size: int = 2,
                        learning_rate: float = 2e-4):
    """Create training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        warmup_steps=10,
        learning_rate=learning_rate,
        fp16=True,  # Use mixed precision
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=50,
        save_total_limit=2,  # Keep only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if wandb.run else None,  # Optional: wandb logging
        dataloader_pin_memory=False,  # Save memory
        remove_unused_columns=False,
        push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
    )
    
    print("✅ Training arguments configured")
    print(f"📊 Total training steps: ~{num_epochs * 40}")  # Approximate
    print(f"⏱️ Estimated training time: ~10-15 minutes")
    
    return training_args

# ============================================================================
# CELL 9: Initialize Training
# ============================================================================

def initialize_training(model, tokenizer, train_dataset, val_dataset, training_args):
    """Initialize the trainer."""
    # Initialize wandb (optional)
    if wandb.run is None:
        wandb.init(
            project="qwen-bi-intent-discovery",
            name=f"qwen2.5-0.5b-bi-intent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": "Qwen/Qwen2.5-0.5B",
                "dataset_size": len(train_dataset) + len(val_dataset),
                "max_length": 2048,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs
            }
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    return trainer

# ============================================================================
# CELL 10: Start Training
# ============================================================================

def train_model(trainer):
    """Start the training process."""
    print("🚀 Starting training...")
    print("⏱️ This will take approximately 10-15 minutes")
    
    # Start training
    trainer.train()
    
    print("✅ Training completed!")
    return trainer

# ============================================================================
# CELL 11: Save the Trained Model
# ============================================================================

def save_model(trainer, model_path: str = "./qwen-bi-intent-model-final"):
    """Save the trained model and configuration."""
    # Save the model
    trainer.save_model(model_path)
    trainer.tokenizer.save_pretrained(model_path)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Save training config
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "training_data_size": len(trainer.train_dataset) + len(trainer.eval_dataset),
        "max_length": 2048,
        "batch_size": trainer.args.per_device_train_batch_size,
        "learning_rate": trainer.args.learning_rate,
        "epochs": trainer.args.num_train_epochs,
        "training_date": datetime.now().isoformat()
    }
    
    with open(f"{model_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Training config saved")
    
    return model_path

# ============================================================================
# CELL 12: Test the Trained Model
# ============================================================================

def load_trained_model(model_path: str):
    """Load the trained model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512):
    """Generate response for a given question."""
    # Create input prompt
    input_text = prompt_template.format(question=question)
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistent outputs
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    response = generated_text[len(input_text):].strip()
    
    return response

def test_model(model_path: str = "./qwen-bi-intent-model-final"):
    """Test the trained model on sample questions."""
    # Test questions
    test_questions = [
        "How many publishers have revenue above $1M?",
        "What is the average age of campaign managers?",
        "Show me the top 5 performing campaigns by engagement rate",
        "Compare revenue between Q1 and Q2 for all publishers"
    ]
    
    print("🧪 Testing the trained model...")
    
    try:
        model, tokenizer = load_trained_model(model_path)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            print("-" * 50)
            
            response = generate_response(model, tokenizer, question)
            print(response)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(response)
                print("✅ Valid JSON output")
            except json.JSONDecodeError:
                print("❌ Invalid JSON output")
            
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running the training cell again or check the model path")

# ============================================================================
# CELL 13: Training Results Analysis
# ============================================================================

def analyze_training_results(trainer):
    """Analyze and display training results."""
    
    # Get training history
    history = trainer.state.log_history
    
    if not history:
        print("❌ No training history available")
        return
    
    print("📊 Training Results Analysis")
    print("=" * 50)
    
    # Extract metrics
    train_losses = [log['loss'] for log in history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in history if 'eval_loss' in log]
    
    if train_losses:
        print(f"📈 Training Loss:")
        print(f"   Start: {train_losses[0]:.4f}")
        print(f"   End: {train_losses[-1]:.4f}")
        print(f"   Improvement: {train_losses[0] - train_losses[-1]:.4f}")
    
    if eval_losses:
        print(f"\n📊 Validation Loss:")
        print(f"   Start: {eval_losses[0]:.4f}")
        print(f"   End: {eval_losses[-1]:.4f}")
        print(f"   Best: {min(eval_losses):.4f}")
    
    # Check for overfitting
    if train_losses and eval_losses:
        final_train_loss = train_losses[-1]
        final_eval_loss = eval_losses[-1]
        
        if final_eval_loss > final_train_loss * 1.2:
            print("\n⚠️ Potential overfitting detected (validation loss > 1.2x training loss)")
        else:
            print("\n✅ No significant overfitting detected")
    
    print(f"\n🎯 Recommendations:")
    if eval_losses and eval_losses[-1] > 2.0:
        print("   • Consider more training epochs")
        print("   • Try different learning rate")
        print("   • Increase dataset size")
    else:
        print("   • Model training looks good!")
        print("   • Consider testing on more examples")
        print("   • Ready for deployment")

# ============================================================================
# CELL 14: Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline."""
    print("🚀 Starting Qwen BI Intent Discovery Training Pipeline")
    print("=" * 60)
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    MAX_LENGTH = 2048
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4
    
    # Step 1: Load data
    print("\n📚 Step 1: Loading training data...")
    training_data = load_training_data('training_data_100_examples.json')
    train_data, val_data = split_data(training_data)
    
    # Step 2: Load prompt template
    print("\n📖 Step 2: Loading prompt template...")
    global prompt_template
    prompt_template = load_prompt_template()
    
    # Step 3: Load model and tokenizer
    print("\n🤖 Step 3: Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Step 4: Setup LoRA
    print("\n🔧 Step 4: Setting up LoRA...")
    model, lora_config = setup_lora(model)
    
    # Step 5: Create datasets
    print("\n📊 Step 5: Creating datasets...")
    train_dataset, val_dataset = create_datasets(train_data, val_data, tokenizer, MAX_LENGTH)
    
    # Step 6: Setup training
    print("\n⚙️ Step 6: Setting up training...")
    training_args = create_training_args(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    trainer = initialize_training(model, tokenizer, train_dataset, val_dataset, training_args)
    
    # Step 7: Train
    print("\n🚀 Step 7: Starting training...")
    trainer = train_model(trainer)
    
    # Step 8: Save model
    print("\n💾 Step 8: Saving model...")
    model_path = save_model(trainer)
    
    # Step 9: Analyze results
    print("\n📊 Step 9: Analyzing results...")
    analyze_training_results(trainer)
    
    # Step 10: Test model
    print("\n🧪 Step 10: Testing model...")
    test_model(model_path)
    
    print("\n🎉 Training pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
