#!/usr/bin/env python3
"""
Debug script to test the formatting function
"""

import json
import sys

def test_formatting():
    """Test the formatting function with sample data"""
    
    # Sample data from your file
    sample_data = [
        {
            "input": "How many heads of the publishers are older than 56 ?",
            "output": {
                "intent": "intents_discovery",
                "discovery_results": [
                    {
                        "step_id": "step_1",
                        "sub_question": "How many heads of the publishers are older than 56 ?",
                        "measures": [
                            {
                                "name": "Heads",
                                "calculation": "Count",
                                "original_phrase": "how many heads"
                            }
                        ],
                        "dimensions": [
                            {
                                "name": "Publisher",
                                "filter_value": None,
                                "original_phrase": "publishers"
                            },
                            {
                                "name": "Age",
                                "filter_value": "older than 56",
                                "original_phrase": "older than 56"
                            }
                        ],
                        "timegrain": None,
                        "timeframe": None,
                        "pattern": None,
                        "segments": [],
                        "breakdowns": [],
                        "unmatched_intents": []
                    }
                ]
            }
        }
    ]
    
    # Test the formatting function
    prompt_template = """# BI Planning & Discovery Agent
You are an AI assistant specialized in analyzing natural language BI questions and breaking them into structured steps for query building.

## Phases
### Phase 1: Planning
- Detect if question is **complex** (multi-step, dependencies, ranking, comparison, or time-based logic).  
- Complexity indicators: "for the X", "top/best/highest/lowest X", "X that are Y", "based on X", "compare X with Y", "X for those Y".  
- If complex:  
  1. Extract BI elements (measures, dimensions, time, filters).  
  2. Break into ordered steps (like CTEs).  
  3. Add post-processing (ranking, sorting, formatting).  
- If simple: skip planning.  

### Phase 2: Discovery
For each question or planning step:  
1. Extract BI concepts (measures, dimensions, timeframes, timegrain, patterns, filters, segments, breakdowns).  
2. Map exact phrases to BI fields (store in `original_phrase`).  
3. Capture **all unmatched terms** in `unmatched_intents` with `phrase`, `type`, and `reason`.  
4. Handle **ambiguity**: If a phrase can mean multiple things, request clarification.  

## Output Format
Respond with a JSON object containing your intent and discovery results.

## Question: {question}

## Response:"""
    
    def format_training_example(example, prompt_template):
        """Test the formatting function"""
        question = example["input"]
        expected_output = json.dumps(example["output"], ensure_ascii=False, indent=2)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(question=question)
        
        # Create the full training text
        training_text = f"{formatted_prompt}\n{expected_output}"
        
        return training_text
    
    # Test the function
    try:
        formatted = format_training_example(sample_data[0], prompt_template)
        print("✅ Formatting successful!")
        print(f"Length: {len(formatted)}")
        print(f"First 200 chars: {formatted[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_formatting()
