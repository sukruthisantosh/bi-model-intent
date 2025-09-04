#!/usr/bin/env python3
"""
Convert extracted_fields.json to GLiNER format
"""

import json
import re

def tokenize_text(text):
    """Use the same tokenization as GLiNER examples"""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def find_entity_in_tokens(tokens, entity_tokens):
    """Find the start and end positions of entity tokens in the main token list"""
    for i in range(len(tokens) - len(entity_tokens) + 1):
        if " ".join(tokens[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
            return i, i + len(entity_tokens) - 1
    return None, None

def convert_extracted_fields_to_gliner(data):
    """Convert extracted fields format to GLiNER format"""
    gliner_data = []
    
    for item in data:
        question = item['question'].strip('"')  # Remove quotes
        tokens = tokenize_text(question)
        
        # Find entities in the tokenized text
        entities = []
        
        # Find MEASURES
        if item['MEASURES'] != 'none':
            measure_tokens = tokenize_text(item['MEASURES'])
            start_pos, end_pos = find_entity_in_tokens(tokens, measure_tokens)
            if start_pos is not None:
                entities.append([start_pos, end_pos, 'MEASURE'])
        
        # Find DIMENSIONS
        if item['DIMENSIONS'] != 'none':
            dim_tokens = tokenize_text(item['DIMENSIONS'])
            start_pos, end_pos = find_entity_in_tokens(tokens, dim_tokens)
            if start_pos is not None:
                entities.append([start_pos, end_pos, 'DIMENSION'])
        
        # Find TIMEFRAME
        if item['TIMEFRAME'] != 'none':
            time_tokens = tokenize_text(item['TIMEFRAME'])
            start_pos, end_pos = find_entity_in_tokens(tokens, time_tokens)
            if start_pos is not None:
                entities.append([start_pos, end_pos, 'TIMEFRAME'])
        
        # Find FILTERS
        if item['FILTERS'] != 'none':
            filter_tokens = tokenize_text(item['FILTERS'])
            start_pos, end_pos = find_entity_in_tokens(tokens, filter_tokens)
            if start_pos is not None:
                entities.append([start_pos, end_pos, 'FILTER'])
        
        # Create GLiNER format item
        gliner_item = {
            "tokenized_text": tokens,
            "ner": entities
        }
        gliner_data.append(gliner_item)
    
    return gliner_data

if __name__ == "__main__":
    print("Converting extracted_fields.json to GLiNER format...")
    
    # Load the extracted fields data
    with open('data/processed/extracted_fields.json', 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    print(f"Loaded {len(extracted_data)} examples from extracted_fields.json")
    
    # Show sample of original data
    print("\nSample original data:")
    print(json.dumps(extracted_data[0], indent=2))
    
    # Convert to GLiNER format
    gliner_data = convert_extracted_fields_to_gliner(extracted_data)
    
    print(f"\nConverted {len(gliner_data)} examples to GLiNER format")
    
    # Show sample converted data
    print("\nSample converted data:")
    print(json.dumps(gliner_data[0], indent=2))
    
    # Check data structure
    print(f"\nData structure check:")
    print(f"First item keys: {list(gliner_data[0].keys())}")
    print(f"Has 'tokenized_text': {'tokenized_text' in gliner_data[0]}")
    print(f"Has 'ner': {'ner' in gliner_data[0]}")
    print(f"Number of entities in first item: {len(gliner_data[0]['ner'])}")
    
    # Check conversion quality
    print(f"\nConversion quality:")
    examples_with_entities = sum(1 for item in gliner_data if len(item['ner']) > 0)
    total_entities = sum(len(item['ner']) for item in gliner_data)
    print(f"Examples with entities: {examples_with_entities}")
    print(f"Total entities found: {total_entities}")
    print(f"Average entities per example: {total_entities / len(gliner_data):.2f}")
    
    # Save converted data
    output_file = 'data/training/gliner_converted_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gliner_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved converted data to: {output_file}")
    print("Conversion complete!")
