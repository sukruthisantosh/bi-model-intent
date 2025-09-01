#!/usr/bin/env python3
"""
NER Annotation Script for Business Intelligence Questions
Uses OpenAI API to annotate questions with entity labels for GLiNER training
Automatically corrects entity indices to match exact substrings in text
"""

import json
import openai
import os
from typing import List, Dict, Any

def load_questions(file_path: str) -> List[str]:
    """Load questions from text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def create_prompt(questions: List[str]) -> str:
    """Create the NER annotation prompt"""
    prompt = """System:
System:
You are a precise NER annotator for Business Intelligence questions. 
Your task is to label entities in questions for GLiNER training. 

Output Rules:
1. Use only the exact substring of the question text.
2. Character indices must be 0-based, end index exclusive.
3. Output JSONL only, with one line per question.
4. Only use these labels:
   - MEASURE: numeric metrics (e.g., sales, revenue, profit)
   - DIMENSION: categorical attributes (e.g., product, region, customer)
   - DIMENSION_VALUE: specific value of a dimension (e.g., Europe, Electronics)
   - TIMEFRAME: relative or explicit time periods (e.g., last year, Q1 2023)
   - TIMEGRAIN: unit of time aggregation (e.g., day, month, quarter)
   - FILTER: any other condition or restriction
5. Do NOT include CALCULATION words (total, average, count, sum, etc.).
6. If a label does not apply, do not include it.
7. Always match the shortest substring that correctly represents the entity.
8. Make sure start and end indices match the substring exactly; do not expand or shrink.

Example Input:
"How many items were sold in Q1 2023?"

Example Output:
{"text":"How many items were sold in Q1 2023?","entities":[[9,14,"MEASURE"],[24,31,"TIMEFRAME"]]}

User:
Now process these questions:
"""
    for i, question in enumerate(questions, 1):
        if i % 100 == 0:
            print(f"  Added {i}/{len(questions)} questions...")
        prompt += f"{i}. {question}\n"
    print(f"  Added all {len(questions)} questions to prompt")
    return prompt

def process_with_openai(prompt: str, api_key: str, model: str = "gpt-4o") -> str:
    """Process the prompt with OpenAI API"""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def parse_jsonl_output(output: str) -> List[Dict[str, Any]]:
    """Parse the JSONL output from OpenAI"""
    results = []
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and 'text' in parsed and 'entities' in parsed:
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results

def fix_entity_indices(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure entity indices match the exact substring in the text.
    Optionally remove calculation words from MEASURE spans (like 'total'/'average').
    """
    text = record["text"]
    new_entities = []
    for start, end, label in record["entities"]:
        predicted = text[start:end]

        # Optional: strip calculation words from MEASURE
        if label == "MEASURE" and predicted.lower().startswith(("total ", "average ")):
            predicted = " ".join(predicted.split(" ")[1:])

        # Find exact occurrence in text
        idx = text.find(predicted)
        if idx == -1:
            idx = text.lower().find(predicted.lower())
        if idx != -1:
            new_entities.append([idx, idx + len(predicted), label])
        else:
            # Fallback: keep original if no match found
            new_entities.append([start, end, label])
    record["entities"] = new_entities
    return record

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} annotated questions to {output_file}")

def main():
    questions_file = "NER/data/processed/remaining_questions.txt"
    output_file = "NER/data/processed/ner_annotated_questions_remaining.jsonl"

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print(f"Loading questions from {questions_file}...")
    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} questions")

    print("Creating annotation prompt...")
    prompt = create_prompt(questions)

    print("Processing with OpenAI API...")
    response = process_with_openai(prompt, api_key)
    if not response:
        print("Failed to get response from OpenAI API")
        return

    print("Parsing results...")
    results = parse_jsonl_output(response)

    print("Fixing entity indices...")
    results = [fix_entity_indices(r) for r in results]
    print(f"Successfully fixed indices for {len(results)} questions")

    save_results(results, output_file)

    print("\nExample annotations:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. {result['text']}")
        if result['entities']:
            for start, end, label in result['entities']:
                entity_text = result['text'][start:end]
                print(f"   [{start}:{end}] {label}: '{entity_text}'")
        else:
            print("   No entities found")

if __name__ == "__main__":
    main()
