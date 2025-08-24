#!/usr/bin/env python3
"""
Interactive Test for Fine-tuned BI Entity Recognition Model
========================================================

This script allows you to interactively test your fine-tuned model
by entering questions and seeing the identified entities.
"""

import json
import os
from openpipe import OpenAI

def test_single_question(question: str, model_name: str = "openpipe:hot-hornets-live"):
    """
    Test a single question with the fine-tuned model.
    
    Args:
        question: The BI question to analyze
        model_name: The OpenPipe model name
    
    Returns:
        The model's response
    """
    # Initialize the OpenPipe client
    client = OpenAI(
        openpipe={"api_key": os.getenv("OPENPIPE_API_KEY")}
    )
    
    # Create the system prompt
    system_prompt = """You are an AI assistant specialized in identifying Business Intelligence (BI) entities from natural language questions.

Your task is to identify and extract the following entity types from BI questions:

1. **Dimensions**: What to group by (e.g., publishers, campaigns, regions, departments)
2. **Measures**: What to measure (e.g., revenue, engagement, users, courses)
3. **Calculations**: How to calculate (e.g., count, sum, average, maximum, minimum)
4. **Filters**: What to filter by (e.g., "older than 56", "above $1M", "alabama")
5. **Time References**: Time-based filters and groupings (e.g., year, "most recently")

Respond with a JSON object containing arrays of identified entities for each type. If no entities of a type are found, use an empty array.

Example format:
{
  "dimensions": ["publisher", "age"],
  "measures": ["heads"],
  "calculations": ["count"],
  "filters": ["older than 56"],
  "time_references": []
}"""
    
    # Make the API call
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        openpipe={
            "tags": {
                "prompt_id": "interactive_test",
                "question_type": "user_input"
            }
        },
    )
    
    return completion.choices[0].message.content

def display_results(question: str, response: str):
    """
    Display the results in a nice format.
    
    Args:
        question: The original question
        response: The model's response
    """
    print(f"\nQuestion: {question}")
    print("=" * 60)
    
    try:
        # Try to parse as JSON
        entities = json.loads(response)
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"{entity_type.capitalize()}:")
                for entity in entity_list:
                    print(f"   • {entity}")
            else:
                print(f"{entity_type.capitalize()}: (none found)")
        
        print("=" * 60)
        
    except json.JSONDecodeError:
        print(f"Raw response (not valid JSON):")
        print(response)
        print("=" * 60)

def main():
    """Main interactive function."""
    print("Interactive BI Entity Recognition Test")
    print("=" * 50)
    print("Enter BI questions and see the identified entities!")
    print("Type 'quit' to exit.")
    print()
    
    # Check if API key is set
    if not os.getenv("OPENPIPE_API_KEY"):
        print("Please set OPENPIPE_API_KEY environment variable")
        print("   export OPENPIPE_API_KEY='your-api-key-here'")
        return
    
    while True:
        try:
            # Get user input
            question = input("\nEnter a BI question: ").strip()
            
            # Check for quit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                continue
            
            print("Analyzing...")
            
            # Test the question
            response = test_single_question(question)
            
            # Display results
            display_results(question, response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
