#!/usr/bin/env python3
"""
Debug script to check GLiNER dataset creation
"""

import json
import torch
from gliner import GLiNER
from gliner.data_processing import GLiNERDataset

# Load the converted data
with open('data/training/gliner_converted_data.json', 'r', encoding='utf-8') as f:
    gliner_data = json.load(f)

print(f"Loaded {len(gliner_data)} examples")

# Show sample data
print("\nSample data:")
print(json.dumps(gliner_data[0], indent=2))

# Load model
print("\nLoading GLiNER model...")
model = GLiNER.from_pretrained("urchade/gliner_small")
print("Model loaded successfully")

# Try to create dataset with just one example
print("\nTesting dataset creation with one example...")
try:
    test_data = [gliner_data[0]]
    dataset = GLiNERDataset(
        examples=test_data,
        config=model.config,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_processor=model.data_processor
    )
    print("Dataset created successfully!")
    print(f"Dataset length: {len(dataset)}")
    
    # Try to access first item
    print("\nTesting dataset access...")
    first_item = dataset[0]
    print(f"First item keys: {list(first_item.keys())}")
    print("Dataset access successful!")
    
except Exception as e:
    print(f"Error creating dataset: {e}")
    print(f"Error type: {type(e)}")
    
    # Check the data structure more carefully
    print("\nDetailed data inspection:")
    print(f"First example keys: {list(gliner_data[0].keys())}")
    print(f"First example 'ner' type: {type(gliner_data[0]['ner'])}")
    print(f"First example 'ner' content: {gliner_data[0]['ner']}")
    print(f"First example 'tokenized_text' type: {type(gliner_data[0]['tokenized_text'])}")
    print(f"First example 'tokenized_text' length: {len(gliner_data[0]['tokenized_text'])}")

# Try with a smaller subset
print("\nTesting with 5 examples...")
try:
    small_data = gliner_data[:5]
    small_dataset = GLiNERDataset(
        examples=small_data,
        config=model.config,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_processor=model.data_processor
    )
    print("Small dataset created successfully!")
    print(f"Small dataset length: {len(small_dataset)}")
    
    # Test accessing items
    for i in range(min(3, len(small_dataset))):
        item = small_dataset[i]
        print(f"Item {i} keys: {list(item.keys())}")
        
except Exception as e:
    print(f"Error with small dataset: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
