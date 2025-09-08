#!/usr/bin/env python3
"""
GLiNER Training Script for CPU with 100 examples
This script trains GLiNER on the first 100 examples from the basic training data
and pushes the results to Hugging Face Hub.
"""

import os
import json
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, login

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset


def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    try:
        # Try to login - you may need to set HF_TOKEN environment variable
        # or run `huggingface-cli login` first
        api = HfApi()
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"Hugging Face authentication failed: {e}")
        print("Please run 'huggingface-cli login' or set HF_TOKEN environment variable")
        return False


def create_cpu_config():
    """Create a configuration optimized for proper GLiNER learning on CPU"""
    config = {
        # Model Configuration
        "model_name": "microsoft/deberta-v3-small",
        "labels_encoder": None,
        "name": "span level gliner",
        "max_width": 12,
        "hidden_size": 768,
        "dropout": 0.3,
        "fine_tune": True,
        "subtoken_pooling": "first",
        "fuse_layers": False,
        "post_fusion_schema": None,
        "span_mode": "token_level",
        
        # Training Parameters - proper learning with CPU constraints
        "num_steps": 1000,  # Enough steps for proper learning
        "train_batch_size": 2,  # Small batch size for CPU
        "eval_every": 200,  # Regular evaluation
        "warmup_ratio": 0.05,  # Standard warmup
        "scheduler_type": "cosine",
        
        # Loss function - standard GLiNER settings
        "loss_alpha": 0.75,
        "loss_gamma": 0,
        "loss_prob_margin": 0,
        "label_smoothing": 0,
        "loss_reduction": "sum",
        
        # Learning Rate and weight decay Configuration - standard GLiNER rates
        "lr_encoder": 1e-5,  # Standard GLiNER learning rate
        "lr_others": 3e-5,   # Standard GLiNER learning rate
        "weight_decay_encoder": 0.1,
        "weight_decay_other": 0.01,
        
        "max_grad_norm": 10.0,
        
        # Directory Paths
        "root_dir": "gliner_logs",
        "train_data": "gliner_training_data_basic.json",
        "val_data_dir": "none",
        
        # Pretrained Model Path
        "prev_path": None,
        
        "save_total_limit": 3,  # Keep checkpoints
        
        # Advanced Training Settings - crucial for GLiNER learning
        "size_sup": -1,
        "max_types": 100,
        "shuffle_types": True,
        "random_drop": True,
        "max_neg_type_ratio": 1,
        "max_len": 512,  # Standard GLiNER max length
        "freeze_token_rep": False,
    }
    return config


def load_and_prepare_data(data_path, num_examples=100):
    """Load and prepare the first N examples from the training data"""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total examples in dataset: {len(data)}")
    
    # Filter out empty examples
    filtered_data = [item for item in data if len(item['tokenized_text']) and len(item['ner'])]
    print(f"Examples with valid data: {len(filtered_data)}")
    
    # Take first N examples
    limited_data = filtered_data[:num_examples]
    print(f"Using first {len(limited_data)} examples for training")
    
    # Shuffle the data
    random.shuffle(limited_data)
    
    # Split into train/test (90/10)
    split_idx = int(len(limited_data) * 0.9)
    train_data = limited_data[:split_idx]
    test_data = limited_data[split_idx:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    return train_data, test_data


def train_model(config_dict, train_data, test_data, output_dir="models/gliner_cpu_100"):
    """Train the GLiNER model"""
    print("Setting up model and training...")
    
    # Create config namespace
    config = argparse.Namespace(**config_dict)
    config.log_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model config
    model_config = GLiNERConfig(**vars(config))
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, add_prefix_space=True)
    
    # Initialize words splitter
    words_splitter = WordsSplitter(model_config.words_splitter_type)
    
    # Create model
    model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)
    
    # Add special tokens if needed
    if not config.labels_encoder:
        model_config.class_token_index = len(tokenizer)
        tokenizer.add_tokens([model_config.ent_token, model_config.sep_token], special_tokens=True)
        model_config.vocab_size = len(tokenizer)
        model.resize_token_embeddings([model_config.ent_token, model_config.sep_token], 
                                    set_class_token_index=False,
                                    add_tokens_to_tokenizer=False)
    
    # Create datasets - use the old data schema (like the original train.py)
    train_dataset = train_data
    test_dataset = test_data
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        focal_loss_gamma=config.loss_gamma,
        focal_loss_alpha=config.loss_alpha,
        focal_loss_prob_margin=config.loss_prob_margin,
        loss_reduction=config.loss_reduction,
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        save_steps=config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=0,  # No multiprocessing for CPU
        use_cpu=True,  # Force CPU usage
        report_to="none",
        bf16=False,  # Disable bf16 for CPU
        fp16=False,  # Disable fp16 for CPU
        logging_steps=10,
        eval_steps=config.eval_every,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Training completed! Model saved to {final_model_path}")
    return final_model_path


def push_to_huggingface(model_path, repo_name="ssuki/gliner-cpu-100-examples"):
    """Push the trained model to Hugging Face Hub"""
    print(f"Pushing model to Hugging Face Hub: {repo_name}")
    
    try:
        from huggingface_hub import HfApi, create_repo, repo_exists
        
        api = HfApi()
        
        # Check if repository exists, create if it doesn't
        if not repo_exists(repo_id=repo_name, repo_type="model"):
            print(f"Repository {repo_name} doesn't exist. Creating it...")
            create_repo(repo_id=repo_name, exist_ok=True, private=False, repo_type="model")
            print(f"Repository {repo_name} created successfully")
        else:
            print(f"Repository {repo_name} already exists")
        
        # Upload the model
        print("Uploading model files...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message="GLiNER model trained on 100 examples with CPU",
            repo_type="model"
        )
        
        print(f"Model successfully pushed to https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"Failed to push to Hugging Face: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train GLiNER on CPU with 100 examples")
    parser.add_argument('--data_path', type=str, default='gliner_training_data_basic.json',
                       help='Path to training data JSON file')
    parser.add_argument('--num_examples', type=int, default=100,
                       help='Number of examples to use for training')
    parser.add_argument('--output_dir', type=str, default='models/gliner_cpu_100',
                       help='Output directory for the trained model')
    parser.add_argument('--push_to_hf', action='store_true',
                       help='Push the trained model to Hugging Face Hub')
    parser.add_argument('--hf_repo_name', type=str, default='ssuki/gliner-cpu-100-examples',
                       help='Hugging Face repository name')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLiNER CPU Training Script")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        return
    
    # Setup Hugging Face authentication if needed
    if args.push_to_hf:
        if not setup_huggingface_auth():
            print("Warning: Hugging Face authentication failed. Model will not be pushed.")
            args.push_to_hf = False
    
    # Create CPU-optimized config
    config = create_cpu_config()
    print(f"Using configuration optimized for proper GLiNER learning on CPU")
    print(f"Training steps: {config['num_steps']}")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Max length: {config['max_len']}")
    print(f"Learning rate (encoder): {config['lr_encoder']}")
    print(f"Learning rate (others): {config['lr_others']}")
    print(f"Evaluation every: {config['eval_every']} steps")
    
    # Load and prepare data
    train_data, test_data = load_and_prepare_data(args.data_path, args.num_examples)
    
    # Train the model
    model_path = train_model(config, train_data, test_data, args.output_dir)
    
    # Push to Hugging Face if requested
    if args.push_to_hf:
        success = push_to_huggingface(model_path, args.hf_repo_name)
        if success:
            print("Model successfully pushed to Hugging Face!")
        else:
            print("Failed to push model to Hugging Face")
    
    print("=" * 60)
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    if args.push_to_hf:
        print(f"Hugging Face repo: https://huggingface.co/{args.hf_repo_name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
