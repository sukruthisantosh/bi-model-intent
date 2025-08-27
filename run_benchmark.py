#!/usr/bin/env python3
"""
Benchmarking Script for Entity Recognition Models
================================================

Easy script to train and compare different models for entity recognition.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_training(model_name, epochs=3, lr=2e-4, batch_size=None, output_dir=None):
    """Run training for a specific model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, "train_entity_recognition.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--lr", str(lr)
    ]
    
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training completed successfully for {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {model_name}: {e}")
        return False

def main():
    """Run benchmarking for different models."""
    print("Entity Recognition Model Benchmarking")
    print("=" * 50)
    
    # Models to benchmark
    models = [
        "qwen-0.5b",
        "qwen-1.5b", 
        "qwen-3b"
    ]
    
    # Training parameters
    epochs = 3
    lr = 2e-4
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./benchmark_results_{timestamp}"
    
    results = []
    
    for model in models:
        output_dir = f"{base_output_dir}/{model}"
        os.makedirs(output_dir, exist_ok=True)
        
        success = run_training(
            model_name=model,
            epochs=epochs,
            lr=lr,
            output_dir=output_dir
        )
        
        results.append({
            "model": model,
            "success": success,
            "output_dir": output_dir
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"{result['model']:15} | {status:10} | {result['output_dir']}")
    
    print(f"\nResults saved to: {base_output_dir}")

if __name__ == "__main__":
    main()
