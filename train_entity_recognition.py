"""
Entity Recognition Training Script
=================================

Simple script to train a model for entity recognition from BI questions.
Change MODEL_NAME to switch between different models.

Expected Output Format:
{
  "dimensions": ["list", "of", "dimensions"],
  "measures": ["list", "of", "measures"], 
  "calculations": ["list", "of", "calculations"],
  "filters": ["list", "of", "filters"],
  "time_references": ["list", "of", "time_references"]
}
"""

import json
import torch
from typing import Dict, List, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from huggingface_hub import login
from training_config import get_config, print_config, TrainingConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "qwen-1.5b"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(config: TrainingConfig) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    with open(config.train_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data

def load_prompt_template(config: TrainingConfig) -> str:
    """Load the entity recognition prompt template."""
    with open(config.prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8):
    """Split data into train/validation sets."""
    train_size = int(train_ratio * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data

# ============================================================================
# DATASET
# ============================================================================

class EntityRecognitionDataset(Dataset):
    """Dataset for entity recognition training."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, prompt_template: str, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Create input prompt
        input_text = self.prompt_template.format(question=example['input'])
        
        # Create expected output
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
# MODEL SETUP
# ============================================================================

def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {config.model.hf_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model.hf_name,
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
    
    print(f"Model loaded: {model.num_parameters():,} parameters")
    return model, tokenizer

def setup_lora(model, config: TrainingConfig):
    """Setup LoRA for efficient fine-tuning."""
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    
    print("LoRA applied successfully")
    model.print_trainable_parameters()
    
    return model, lora_config

# ============================================================================
# TRAINING
# ============================================================================

def create_datasets(train_data, val_data, tokenizer, prompt_template, config: TrainingConfig):
    """Create training and validation datasets."""
    train_dataset = EntityRecognitionDataset(train_data, tokenizer, prompt_template, config.model.max_length)
    val_dataset = EntityRecognitionDataset(val_data, tokenizer, prompt_template, config.model.max_length)
    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset, config: TrainingConfig):
    """Train the model."""
    print("Starting training...")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.model.batch_size,
        per_device_eval_batch_size=config.model.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"Training completed! Model saved to {config.output_dir}")
    return trainer

# ============================================================================
# TESTING
# ============================================================================

def test_model(model, tokenizer, test_questions: List[str], config: TrainingConfig):
    """Test the trained model on sample questions."""
    print("\nTesting model on sample questions...")
    
    prompt_template = load_prompt_template(config)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        input_text = prompt_template.format(question=question)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.model.max_length)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        print(f"Response: {response}")
        
        try:
            parsed = json.loads(response)
            print(f"Valid JSON: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError:
            print("Invalid JSON format")

# ============================================================================
# HUGGINGFACE HUB
# ============================================================================

def push_to_hub(model, tokenizer, config: TrainingConfig):
    """Push the trained model to HuggingFace Hub."""
    print("\nPushing model to HuggingFace Hub...")
    
    repo_name = f"ssuki/{config.model.name}-entity-recognition"
    
    try:
        model.push_to_hub(repo_name, private=False)
        tokenizer.push_to_hub(repo_name, private=False)
        print(f"Successfully pushed to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        print("Make sure you're logged in: huggingface-cli login")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Get configuration
    config = get_config(model_name=MODEL_NAME)
    
    # Check HuggingFace login
    try:
        from huggingface_hub import whoami
        username = whoami()
        print(f"Logged in to HuggingFace as: {username}")
    except Exception:
        print("Warning: Not logged in to HuggingFace Hub")
        print("To push model after training, run: huggingface-cli login")
    
    print("Starting Entity Recognition Training Pipeline")
    print_config(config)
    
    # Check GPU
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    data = load_training_data(config)
    train_data, val_data = split_data(data)
    
    # Load prompt template
    prompt_template = load_prompt_template(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    model, lora_config = setup_lora(model, config)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(train_data, val_data, tokenizer, prompt_template, config)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    # Test model
    test_questions = [
        "How many heads of the publishers are older than 56?",
        "What is the average revenue of departments?",
        "List the names of publishers created in California"
    ]
    
    test_model(model, tokenizer, test_questions, config)
    
    # Push to HuggingFace Hub
    push_to_hub(model, tokenizer, config)
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main()
