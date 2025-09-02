#!/usr/bin/env python3
"""
SpaCy Index Generator for Extracted Fields
Automatically generates correct character indices for entity spans using SpaCy tokenization
"""

import json
import spacy
from typing import List, Dict, Any, Tuple
import re

def load_extracted_fields(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load extracted fields from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        return data[:limit]
    return data

def clean_question_text(question: str) -> str:
    """Clean question text by removing quotes and extra whitespace"""
    # Remove surrounding quotes and clean whitespace
    cleaned = question.strip().strip('"').strip("'")
    return cleaned

def find_entity_span(text: str, entity_text: str, label: str) -> List[List]:
    """Find all possible spans for an entity text in the question"""
    if not entity_text or entity_text.lower() == "none":
        return []
    
    # Clean entity text
    entity_text = entity_text.strip()
    
    # Convert both to lowercase for case-insensitive matching
    text_lower = text.lower()
    entity_lower = entity_text.lower()
    
    # Find all occurrences of the entity text (case-insensitive)
    results = []
    start = 0
    
    while True:
        # Find the next occurrence in lowercase
        pos = text_lower.find(entity_lower, start)
        if pos == -1:
            break
        
        # Verify it's a word boundary match (not part of another word)
        if is_word_boundary(text, pos, pos + len(entity_text)):
            # Use the original case from the question text for the span
            actual_span = text[pos:pos + len(entity_text)]
            results.append([pos, pos + len(entity_text), label])
        
        start = pos + 1
    
    return results

def is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if the span represents a complete word boundary"""
    # Check if start is at beginning or after non-alphanumeric
    start_ok = start == 0 or not text[start-1].isalnum()
    
    # Check if end is at end or before non-alphanumeric
    end_ok = end == len(text) or not text[end].isalnum()
    
    return start_ok and end_ok

def extract_entities_from_fields(fields: Dict[str, Any]) -> List[List]:
    """Extract all entities from a fields record"""
    question = clean_question_text(fields['question'])
    entities = []
    
    # Process each field type
    field_mappings = {
        'MEASURES': 'MEASURE',
        'DIMENSIONS': 'DIMENSION', 
        'TIMEFRAME': 'TIMEFRAME',
        'FILTERS': 'FILTER'
    }
    
    for field_name, label in field_mappings.items():
        if field_name in fields:
            field_value = fields[field_name]
            if field_value and field_value.lower() != "none":
                # Handle special cases like "Loyalty Program = true"
                if "=" in field_value:
                    # Extract just the entity part before the equals
                    entity_part = field_value.split("=")[0].strip()
                    if entity_part:
                        spans = find_entity_span(question, entity_part, label)
                        entities.extend(spans)
                else:
                    spans = find_entity_span(question, field_value, label)
                    entities.extend(spans)
    
    # Sort entities by start position
    entities.sort(key=lambda x: x[0])
    
    return entities

def process_extracted_fields(fields_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process all extracted fields and generate entity annotations"""
    results = []
    
    for i, fields in enumerate(fields_list, 1):
        print(f"Processing example {i}: {fields['question'][:80]}...")
        
        # Extract entities with correct indices
        entities = extract_entities_from_fields(fields)
        
        # Create the result record
        result = {
            'text': clean_question_text(fields['question']),
            'entities': entities
        }
        
        results.append(result)
        
        # Show what was found
        print(f"  Found {len(entities)} entities:")
        for start, end, label in entities:
            entity_text = result['text'][start:end]
            print(f"    [{start}:{end}] {label}: '{entity_text}'")
        print()
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} annotated questions to {output_file}")

def main():
    # Configuration
    input_file = "NER/data/processed/extracted_fields.json"
    output_file = "NER/data/training/spacy_generated_annotations.jsonl"
    limit = 10  # Start with first 10 examples
    
    print("SpaCy Index Generator for Extracted Fields")
    print("=" * 50)
    
    try:
        # Load extracted fields
        print(f"Loading extracted fields from {input_file}...")
        fields_list = load_extracted_fields(input_file, limit)
        print(f"Loaded {len(fields_list)} examples (limited to first {limit})")
        
        # Process fields and generate annotations
        print(f"\nProcessing {len(fields_list)} examples with SpaCy...")
        results = process_extracted_fields(fields_list)
        
        # Save results
        print(f"\nSaving results...")
        save_results(results, output_file)
        
        # Summary
        print(f"\n{'='*50}")
        print("SUMMARY:")
        print(f"Examples processed: {len(results)}")
        total_entities = sum(len(r['entities']) for r in results)
        print(f"Total entities found: {total_entities}")
        print(f"Output saved to: {output_file}")
        print(f"{'='*50}")
        
        # Show validation example
        if results:
            print(f"\nValidation - First example:")
            example = results[0]
            print(f"Question: {example['text']}")
            print("Entities:")
            for start, end, label in example['entities']:
                entity_text = example['text'][start:end]
                print(f"  [{start}:{end}] {label}: '{entity_text}'")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
