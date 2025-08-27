#!/usr/bin/env python3
"""
Training Configuration for Entity Recognition
============================================

Easy model switching and minimal hyperparameters for benchmarking.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration for different models."""
    name: str
    hf_name: str
    max_length: int = 1024
    batch_size: int = 4

@dataclass
class TrainingConfig:
    """Minimal training configuration for benchmarking."""
    # Model
    model: ModelConfig
    
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 50
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Data
    train_data_path: str = "./training_data_new/training_data_entity_recognition.json"
    prompt_path: str = "./entity_recognition_prompt.txt"
    output_dir: str = "./entity_recognition_model"
    
    # Logging
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

# Different models
MODELS = {
    "qwen-1.5b": ModelConfig(
        name="qwen-1.5b",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_length=1024,
        batch_size=4
    ),
    "qwen-0.5b": ModelConfig(
        name="qwen-0.5b", 
        hf_name="Qwen/Qwen2.5-0.5B",
        max_length=1024,
        batch_size=8
    )
}

def get_config(model_name: str = "qwen-1.5b", **kwargs) -> TrainingConfig:
    """Get training configuration for a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_name]
    
    # Create training config with model
    config = TrainingConfig(model=model_config)
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    
    return config

def list_models():
    """List all available models."""
    print("Available models:")
    for name, model in MODELS.items():
        print(f"  {name}: {model.hf_name}")
    print()

def print_config(config: TrainingConfig):
    """Print configuration details."""
    print("Training Configuration:")
    print(f"  Model: {config.model.name} ({config.model.hf_name})")
    print(f"  Max Length: {config.model.max_length}")
    print(f"  Batch Size: {config.model.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LoRA r: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Output Dir: {config.output_dir}")
