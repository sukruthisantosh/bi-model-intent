#!/usr/bin/env python3
"""
Create Remaining Questions Script
Extracts questions that are NOT in the ner_annotated_questions_corrected.jsonl file
from the original questions_text_only.txt file
"""

import json
import sys
from typing import Set, List

def load_annotated_questions(file_path: str) -> Set[str]:
    """Load questions that have already been annotated"""
    annotated_questions = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                annotated_questions.add(data['text'])
            except json.JSONDecodeError:
                continue
    
    return annotated_questions

def load_all_questions(file_path: str) -> List[str]:
    """Load all questions from the original text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def find_remaining_questions(all_questions: List[str], annotated_questions: Set[str]) -> List[str]:
    """Find questions that are not in the annotated set"""
    remaining = []
    
    for question in all_questions:
        if question not in annotated_questions:
            remaining.append(question)
    
    return remaining

def save_remaining_questions(questions: List[str], output_file: str):
    """Save the remaining questions to a new file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')
    
    print(f"Saved {len(questions)} remaining questions to {output_file}")

def main():
    # File paths
    annotated_file = "NER/ner_annotated_questions_corrected.jsonl"
    all_questions_file = "NER/data/processed/questions_text_only.txt"
    output_file = "NER/data/processed/remaining_questions.txt"
    
    try:
        # Load annotated questions
        print(f"Loading annotated questions from {annotated_file}...")
        annotated_questions = load_annotated_questions(annotated_file)
        print(f"Found {len(annotated_questions)} annotated questions")
        
        # Load all questions
        print(f"Loading all questions from {all_questions_file}...")
        all_questions = load_all_questions(all_questions_file)
        print(f"Found {len(all_questions)} total questions")
        
        # Find remaining questions
        print("Finding remaining questions...")
        remaining_questions = find_remaining_questions(all_questions, annotated_questions)
        print(f"Found {len(remaining_questions)} remaining questions")
        
        # Save remaining questions
        save_remaining_questions(remaining_questions, output_file)
        
        # Show summary
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"Total questions: {len(all_questions)}")
        print(f"Annotated questions: {len(annotated_questions)}")
        print(f"Remaining questions: {len(remaining_questions)}")
        print(f"{'='*60}")
        
        # Show some examples of remaining questions
        if remaining_questions:
            print("\nExamples of remaining questions:")
            for i, question in enumerate(remaining_questions[:5], 1):
                print(f"{i}. {question}")
            if len(remaining_questions) > 5:
                print(f"... and {len(remaining_questions) - 5} more")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
