#!/usr/bin/env python3
"""
Test LLM Processing
==================

This script tests the LLM processing with a few sample questions to ensure
it works correctly before processing the full dataset.
"""

import os
import json
import openai
from typing import Dict, Any

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template"""
    with open(prompt_path, "r") as f:
        return f.read()

def build_prompt(prompt_template: str, question: str) -> str:
    """Build the full prompt for a question"""
    return prompt_template.replace("{{ context.current_question }}", question)

def process_question(question: str, prompt_template: str) -> Dict[str, Any]:
    """Process a single question through the LLM"""
    prompt = build_prompt(prompt_template, question)
    
    print(f"\n{'='*60}")
    print(f"Processing: {question}")
    print(f"{'='*60}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Using cheaper model for cost optimization
            messages=[
                {"role": "system", "content": "You are a helpful assistant for BI intent discovery. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        print("✅ Successfully processed!")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        return {
            "input": question,
            "output": result
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    """Test the LLM processing with sample questions"""
    print("LLM Processing Test")
    print("=" * 50)
    
    # Load prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompts/prompt.txt")
    prompt_template = load_prompt_template(prompt_path)
    print(f"Prompt template loaded ({len(prompt_template)} characters)")
    
    # Test questions (representing different types of issues found in the data)
    test_questions = [
        # Simple question
        "How many heads of the publishers are older than 56?",
        
        # Question with ordering (should be complex)
        "List the name, created state and age of the heads of publishers ordered by age.",
        
        # Question with revenue (should not be mapped to budget)
        "What are the maximum and minimum revenue of the publishers?",
        
        # Complex question with multiple steps
        "Compare revenue between Q1 and Q2 for top performing publishers",
        
        # Ambiguous question (should trigger request_human_input)
        "Show me performance for ABC"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\nTest {i}/{len(test_questions)}")
        result = process_question(question, prompt_template)
        if result:
            results.append(result)
    
    # Save test results
    output_file = os.path.join(os.path.dirname(__file__), "../test_llm_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"Test Results Summary")
    print(f"{'='*60}")
    print(f"Processed: {len(results)}/{len(test_questions)} questions successfully")
    print(f"Results saved to: {output_file}")
    
    # Quick validation
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Input: {result['input']}")
        print(f"Intent: {result['output'].get('intent', 'N/A')}")
        if 'discovery_results' in result['output']:
            steps = len(result['output']['discovery_results'])
            print(f"Steps: {steps}")
        print("-" * 40)

if __name__ == "__main__":
    main()
