#!/usr/bin/env python3
"""
NER Annotation Validation Script
Shows each annotated question with highlighted entity spans for manual validation
"""

import json
import sys
from typing import List, Dict, Any

def load_annotations(file_path: str) -> List[Dict[str, Any]]:
    """Load the annotated questions from JSONL file"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                results.append(parsed)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line}")
                continue
    return results

def display_question_with_entities(question_data: Dict[str, Any], question_num: int):
    """Display a single question with highlighted entity spans"""
    text = question_data['text']
    entities = question_data.get('entities', [])
    
    print(f"\n{'='*80}")
    print(f"QUESTION {question_num}: {text}")
    print(f"{'='*80}")
    
    if not entities:
        print("  No entities found")
        return
    
    # Sort entities by start position to display in order
    sorted_entities = sorted(entities, key=lambda x: x[0])
    
    print(f"Found {len(entities)} entities:")
    print()
    
    for i, (start, end, label) in enumerate(sorted_entities, 1):
        # Extract the entity text
        entity_text = text[start:end]
        
        # Create a visual representation with the entity highlighted
        before = text[:start]
        after = text[end:]
        
        print(f"Entity {i}: [{start}:{end}] {label}")
        print(f"  Text: '{entity_text}'")
        print(f"  Context: '{before}'[{entity_text}]{after}'")
        print()
    
    # Show the full text with entity positions marked
    print("Full text with entity positions:")
    print(text)
    
    # Mark entity positions below the text
    position_markers = [' '] * len(text)
    for start, end, label in sorted_entities:
        # Mark the start and end of each entity
        if start < len(position_markers):
            position_markers[start] = '^'
        if end - 1 < len(position_markers):
            position_markers[end - 1] = '^'
    
    print(''.join(position_markers))
    
    # Show entity labels below
    label_line = [' '] * len(text)
    for start, end, label in sorted_entities:
        # Center the label above the entity span
        label_start = start + (end - start) // 2 - len(label) // 2
        label_start = max(0, label_start)
        label_end = min(len(text), label_start + len(label))
        
        for i in range(label_start, label_end):
            if i < len(label_line):
                label_line[i] = label[i - label_start]
    
    print(''.join(label_line))

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 validate_ner_annotations.py <annotations_file>")
        print("Example: python3 validate_ner_annotations.py data/processed/ner_annotated_questions_corrected.jsonl")
        sys.exit(1)
    
    annotations_file = sys.argv[1]
    
    try:
        annotations = load_annotations(annotations_file)
        print(f"Loaded {len(annotations)} annotated questions from {annotations_file}")
    except FileNotFoundError:
        print(f"Error: File {annotations_file} not found")
        sys.exit(1)
    
    if not annotations:
        print("No annotations found in the file")
        sys.exit(1)
    
    print(f"\nStarting validation of {len(annotations)} questions...")
    print("For each question, you'll see:")
    print("- The original question text")
    print("- Each entity with its start/end indices and label")
    print("- The entity text in context")
    print("- Visual markers showing entity positions")
    print("\nPress Enter after reviewing each question to continue...")
    
    for i, annotation in enumerate(annotations, 1):
        display_question_with_entities(annotation, i)
        
        if i < len(annotations):
            input(f"\nPress Enter to continue to question {i+1}...")
    
    print(f"\n{'='*80}")
    print(f"Validation complete! Reviewed {len(annotations)} questions.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
