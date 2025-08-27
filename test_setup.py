#!/usr/bin/env python3
"""
Test Setup for Entity Recognition Training
=========================================

Quick test to verify everything is working correctly.
"""

import sys
import json
from training_config import list_models, get_config, print_config

def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    
    # List available models
    list_models()
    
    # Test getting config for different models
    models_to_test = ["qwen-1.5b", "qwen-0.5b"]
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}:")
        config = get_config(model_name)
        print_config(config)
    
    print("Configuration system working correctly!")

def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    try:
        with open("./training_data_new/training_data_entity_recognition.json", 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} training examples")
        
        # Show sample
        print("Sample example:")
        print(json.dumps(data[0], indent=2))
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    return True

def test_prompt_loading():
    """Test prompt loading."""
    print("\nTesting prompt loading...")
    
    try:
        with open("./entity_recognition_prompt.txt", 'r') as f:
            prompt = f.read()
        print(f"Successfully loaded prompt ({len(prompt)} characters)")
        print("Prompt preview:")
        print(prompt[:200] + "...")
        
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing Entity Recognition Training Setup")
    print("=" * 50)
    
    # Test configuration
    test_config()
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test prompt loading
    prompt_ok = test_prompt_loading()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Configuration: OK")
    print(f"Data loading: {'OK' if data_ok else 'FAILED'}")
    print(f"Prompt loading: {'OK' if prompt_ok else 'FAILED'}")
    
    if data_ok and prompt_ok:
        print("\nAll tests passed! Ready to train.")
        print("\nTo start training:")
        print("  python train_entity_recognition.py --model qwen-1.5b")
        print("\nTo see all available models:")
        print("  python train_entity_recognition.py --list-models")
        print("\nTo run benchmarking:")
        print("  python run_benchmark.py")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
