#!/usr/bin/env python3
"""
Extract exactly 50 questions from each domain from the generated BI questions
"""

import json
import os
from collections import defaultdict
import random

def load_questions(file_path):
    """Load questions from the generated file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_balanced_questions(questions, questions_per_domain=50):
    """Extract exactly 50 questions from each domain"""
    
    # Group questions by domain
    domain_questions = defaultdict(list)
    
    for question in questions:
        domain = question.get("DOMAIN", "unknown")
        if domain != "unknown":
            domain_questions[domain].append(question)
    
    # Extract exactly 50 from each domain
    balanced_questions = []
    
    for domain, domain_qs in domain_questions.items():
        print(f"Domain: {domain} - Found {len(domain_qs)} questions")
        
        if len(domain_qs) >= questions_per_domain:
            # Randomly sample 50 questions from this domain
            selected = random.sample(domain_qs, questions_per_domain)
            balanced_questions.extend(selected)
            print(f"  → Selected {len(selected)} questions")
        else:
            # Use all available questions (less than 50)
            balanced_questions.extend(domain_qs)
            print(f"  → Used all {len(domain_qs)} questions (less than {questions_per_domain})")
    
    return balanced_questions

def extract_question_text_only(questions):
    """Extract only the question text, removing all other fields"""
    question_texts = []
    
    for question in questions:
        # Get just the question text
        question_text = question.get("question", "")
        if question_text:
            # Clean up the question text (remove quotes if present)
            question_text = question_text.strip('"')
            question_texts.append(question_text)
    
    return question_texts

def save_questions(questions, output_file):
    """Save the balanced questions to a new file"""
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"\nSaved {len(questions)} balanced questions to {output_file}")

def save_questions_text_only(question_texts, output_file):
    """Save only the question texts to a file"""
    with open(output_file, 'w') as f:
        for question in question_texts:
            f.write(question + '\n')
    
    print(f"\nSaved {len(question_texts)} question texts to {output_file}")

def main():
    # Find the most recent raw questions file
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} directory not found")
        return
    
    # Get all raw question files
    raw_files = [f for f in os.listdir(raw_dir) if f.startswith("bi_questions_raw_") and f.endswith(".json")]
    
    if not raw_files:
        print("No raw question files found")
        return
    
    # Sort by timestamp (newest first)
    raw_files.sort(reverse=True)
    latest_file = raw_files[0]
    input_path = os.path.join(raw_dir, latest_file)
    
    print(f"Loading questions from: {input_path}")
    
    try:
        questions = load_questions(input_path)
        print(f"Loaded {len(questions)} total questions")
        
        # Extract balanced questions (50 per domain)
        balanced_questions = extract_balanced_questions(questions, questions_per_domain=50)
        
        # Extract only the question text
        question_texts = extract_question_text_only(balanced_questions)
        
        # Create output filenames
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = latest_file.replace("bi_questions_raw_", "").replace(".json", "")
        
        # Save both formats
        json_output_file = os.path.join(output_dir, f"balanced_questions_50_per_domain_{timestamp}.json")
        text_output_file = os.path.join(output_dir, f"questions_text_only_{timestamp}.txt")
        
        # Save the full balanced questions (JSON)
        save_questions(balanced_questions, json_output_file)
        
        # Save only the question texts (plain text, one per line)
        save_questions_text_only(question_texts, text_output_file)
        
        # Print summary
        print("\n" + "="*50)
        print("BALANCED QUESTIONS SUMMARY")
        print("="*50)
        
        domain_counts = defaultdict(int)
        for q in balanced_questions:
            domain = q.get("DOMAIN", "unknown")
            domain_counts[domain] += 1
        
        for domain, count in sorted(domain_counts.items()):
            print(f"{domain:20}: {count:3} questions")
        
        print(f"\nTotal questions: {len(balanced_questions)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
