#!/usr/bin/env python3
"""
Test Fine-tuned OpenPipe Model for BI Entity Recognition
======================================================

This script demonstrates how to use your fine-tuned model to identify
BI entities from natural language questions.
"""

import json
import os
from openpipe import OpenAI

def test_bi_entity_recognition(question: str, model_name: str = "openpipe:hot-hornets-live"):
    """
    Test the fine-tuned model on a BI question.
    
    Args:
        question: The BI question to analyze
        model_name: The OpenPipe model name
    
    Returns:
        The model's response with identified entities
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
        temperature=0,  # Low temperature for consistent outputs
        openpipe={
            "tags": {
                "prompt_id": "bi_entity_recognition",
                "question_type": "bi_query"
            }
        },
    )
    
    return completion.choices[0].message.content

def parse_entity_response(response: str):
    """
    Parse the model's JSON response and format it nicely.
    
    Args:
        response: The model's response string
    
    Returns:
        Parsed entities dictionary
    """
    try:
        # Try to parse as JSON
        entities = json.loads(response)
        return entities
    except json.JSONDecodeError:
        print(f"⚠️ Could not parse response as JSON: {response}")
        return None

def display_entities(question: str, entities: dict):
    """
    Display the identified entities in a nice format.
    
    Args:
        question: The original question
        entities: The parsed entities
    """
    print(f"🔍 Question: {question}")
    print("=" * 60)
    
    if not entities:
        print("No entities found or invalid response")
        return
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"📊 {entity_type.capitalize()}:")
            for entity in entity_list:
                print(f"   • {entity}")
        else:
            print(f"📊 {entity_type.capitalize()}: (none found)")
    
    print("=" * 60)

def main():
    """Main function to test the fine-tuned model."""
    print("Testing Fine-tuned BI Entity Recognition Model")
    print("=" * 60)
    
    # Check if API key is set
    if not os.getenv("OPENPIPE_API_KEY"):
        print("Please set OPENPIPE_API_KEY environment variable")
        print("   export OPENPIPE_API_KEY='your-api-key-here'")
        return
    
    # Test questions
    test_questions = [
        "How many heads of the publishers are older than 56?",
        "List the name, created state and age of the heads of publishers ordered by age.",
        "What is the average revenue of campaigns in Q1?",
        "Show me the top 5 performing departments by engagement rate",
        "Compare revenue between Q1 and Q2 for all publishers"
    ]
    
    print(f"Testing {len(test_questions)} questions...")
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}/{len(test_questions)}")
        
        try:
            # Get model response
            response = test_bi_entity_recognition(question)
            
            # Parse the response
            entities = parse_entity_response(response)
            
            # Display results
            display_entities(question, entities)
            
        except Exception as e:
            print(f"Error testing question {i}: {e}")
        
        print()  # Add spacing between tests
    
    print("🎉 Testing completed!")
    print("\n💡 You can now use this model to identify BI entities from any question!")

if __name__ == "__main__":
    main()
