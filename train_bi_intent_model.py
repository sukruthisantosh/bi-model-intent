#!/usr/bin/env python3
"""
BI Intent Discovery Model Training Script
Fine-tunes Qwen model to perform BI planning and discovery tasks
"""

import os
import json
import torch
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from huggingface_hub import login
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_name: str = "bi-intent-discovery-qwen"
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # Data settings
    train_data_path: str = "training_data_500_examples.json"
    prompt_template_path: str = "resources/prompts/prompt_new.txt"
    
    # Output settings
    output_dir: str = "./trained_model"
    save_to_hf: bool = True
    hf_username: str = None
    
    # Hardware settings
    use_4bit: bool = True
    use_8bit: bool = False
    use_flash_attention: bool = True

class BIIntentTrainer:
    """Trainer for BI Intent Discovery model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_prompt_template(self) -> str:
        """Load the prompt template"""
        try:
            with open(self.config.prompt_template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Prompt template not found at {self.config.prompt_template_path}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Default prompt template if file not found"""
        return """# BI Planning & Discovery Agent
You are an AI assistant specialized in analyzing natural language BI questions and breaking them into structured steps for query building.

## Phases
### Phase 1: Planning
- Detect if question is **complex** (multi-step, dependencies, ranking, comparison, or time-based logic).  
- Complexity indicators: "for the X", "top/best/highest/lowest X", "X that are Y", "based on X", "compare X with Y", "X for those Y".  
- If complex:  
  1. Extract BI elements (measures, dimensions, time, filters).  
  2. Break into ordered steps (like CTEs).  
  3. Add post-processing (ranking, sorting, formatting).  
- If simple: skip planning.  

### Phase 2: Discovery
For each question or planning step:  
1. Extract BI concepts (measures, dimensions, timeframes, timegrain, patterns, filters, segments, breakdowns).  
2. Map exact phrases to BI fields (store in `original_phrase`).  
3. Capture **all unmatched terms** in `unmatched_intents` with `phrase`, `type`, and `reason`.  
4. Handle **ambiguity**: If a phrase can mean multiple things, request clarification.  

## Output Format
Respond with a JSON object containing your intent and discovery results.

## Question: {question}

## Response:"""
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load and preprocess training data"""
        logger.info(f"Loading training data from {self.config.train_data_path}")
        
        try:
            with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data not found at {self.config.train_data_path}")
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def format_training_example(self, example: Dict[str, Any], prompt_template: str) -> str:
        """Format a training example into the model's expected format"""
        question = example["input"]
        expected_output = json.dumps(example["output"], ensure_ascii=False, indent=2)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(question=question)
        
        # Create the full training text
        training_text = f"{formatted_prompt}\n{expected_output}"
        
        return training_text
    
    def prepare_dataset(self, data: List[Dict[str, Any]], prompt_template: str) -> Dataset:
        """Prepare the dataset for training"""
        logger.info("Preparing dataset...")
        
        formatted_examples = []
        for example in data:
            try:
                formatted_text = self.format_training_example(example, prompt_template)
                formatted_examples.append({"text": formatted_text})
            except Exception as e:
                logger.warning(f"Error formatting example: {e}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_examples)} examples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if self.config.use_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            })
        elif self.config.use_8bit:
            model_kwargs["load_in_8bit"] = True
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
    
    def setup_training(self, dataset: Dataset):
        """Setup the training configuration"""
        logger.info("Setting up training...")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset.select(range(min(100, len(tokenized_dataset)))),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Training setup completed")
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training...")
        
        # Load prompt template
        prompt_template = self.load_prompt_template()
        
        # Load and prepare data
        raw_data = self.load_training_data()
        dataset = self.prepare_dataset(raw_data, prompt_template)
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup training
        self.setup_training(dataset)
        
        # Start training
        logger.info("Training started...")
        train_result = self.trainer.train()
        
        # Save the model
        logger.info("Saving model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        with open(os.path.join(self.config.output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return train_result
    
    def save_to_huggingface(self):
        """Save the trained model to Hugging Face Hub"""
        if not self.config.save_to_hf:
            logger.info("Skipping Hugging Face upload (save_to_hf=False)")
            return
        
        if not self.config.hf_username:
            logger.warning("HF username not provided, skipping upload")
            return
        
        try:
            # Login to Hugging Face
            login()
            
            # Model name for HF
            model_name = f"{self.config.hf_username}/{self.config.model_name}"
            
            logger.info(f"Uploading model to Hugging Face: {model_name}")
            
            # Push model and tokenizer
            self.model.push_to_hub(model_name)
            self.tokenizer.push_to_hub(model_name)
            
            # Create model card
            self._create_model_card(model_name)
            
            logger.info(f"Model successfully uploaded to: https://huggingface.co/{model_name}")
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")
    
    def _create_model_card(self, model_name: str):
        """Create a model card for the uploaded model"""
        model_card = f"""---
language:
- en
tags:
- bi-intent-discovery
- business-intelligence
- question-analysis
- structured-output
license: mit
---

# BI Intent Discovery Model

This model is fine-tuned from Qwen2.5-7B-Instruct to perform Business Intelligence (BI) intent discovery tasks.

## Model Description

The model analyzes natural language questions about business intelligence data and breaks them down into structured steps for query building. It performs two main phases:

1. **Planning Phase**: Detects complex questions and breaks them into ordered steps
2. **Discovery Phase**: Extracts BI concepts (measures, dimensions, timeframes, etc.) from questions

## Training Data

- 500 examples of BI questions with structured outputs
- Covers various complexity levels from simple to multi-step queries
- Includes examples with ambiguity handling and unmatched intent capture

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load model
model_name = "{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Example usage
question = "Show me total sales by region for the last quarter"
# Format with your prompt template and generate response
```

## Output Format

The model outputs structured JSON containing:
- Intent classification
- Discovery results with measures, dimensions, timeframes
- Unmatched intents for ambiguous terms
- Step-by-step breakdown for complex queries

## Training Configuration

- Base Model: Qwen2.5-7B-Instruct
- Training Examples: 500
- Epochs: {self.config.num_epochs}
- Learning Rate: {self.config.learning_rate}
- Max Sequence Length: {self.config.max_seq_length}
"""
        
        # Save model card
        card_path = os.path.join(self.config.output_dir, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        
        # Upload model card
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=model_name,
                repo_type="model"
            )
        except Exception as e:
            logger.warning(f"Could not upload model card: {e}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train BI Intent Discovery Model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model to fine-tune")
    parser.add_argument("--train-data", default="training_data_500_examples.json",
                       help="Path to training data")
    parser.add_argument("--prompt-template", default="resources/prompts/prompt_new.txt",
                       help="Path to prompt template")
    parser.add_argument("--output-dir", default="./trained_model",
                       help="Output directory for trained model")
    parser.add_argument("--hf-username", help="Hugging Face username for model upload")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--no-hf-upload", action="store_true", help="Skip HF upload")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        base_model=args.base_model,
        train_data_path=args.train_data,
        prompt_template_path=args.prompt_template,
        output_dir=args.output_dir,
        hf_username=args.hf_username,
        save_to_hf=not args.no_hf_upload,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Initialize trainer
    trainer = BIIntentTrainer(config)
    
    # Train model
    train_result = trainer.train()
    
    # Save to Hugging Face if requested
    if config.save_to_hf and config.hf_username:
        trainer.save_to_huggingface()
    
    logger.info("Training process completed successfully!")

if __name__ == "__main__":
    main()
