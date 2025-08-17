#!/usr/bin/env python3
"""
Interactive Chat with Trained Qwen BI Intent Model
=================================================

This script loads your trained Qwen model and provides an interactive chat interface
to test BI intent discovery capabilities.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from typing import Dict, Any

# Load the prompt template
def load_prompt_template() -> str:
    """Load the prompt template for BI intent discovery."""
    prompt_template = """ # BI Planning & Discovery Agent
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
   - Include business entities, descriptors, product categories, customer types, actions, etc.  
   - Better to over-capture than miss.  
4. Handle **ambiguity**:  
   - If a phrase can mean multiple things, request clarification.  
   - Provide reasoning + clear options for what the phrase could mean.  

## Rules
- Always preserve exact phrase in `original_phrase`.  
- If no timeframe/pattern, leave null/empty.  
- Multiple filter values → single string (`"Desktop, Mobile"`).  
- Discovery steps = same number as planning steps.  
- When identifying BI concepts in a step, **ignore references to outputs of prior steps**.  

## Output Format
```json
{
 "intent": "intents_discovery",
 "discovery_results": [
   {
     "step_id": "step_1",
     "sub_question": "...",
     "measures": [],
     "dimensions": [],
     "timegrain": null,
     "timeframe": null,
     "pattern": null,
     "segments": [],
     "breakdowns": [],
     "unmatched_intents": []
   }
 ]
}
```

## Examples
{question}

Example 1: Simple
Q: "Show me total sales"
```json
{
"intent": "intents_discovery",
"discovery_results": [
{
"step_id": "step_1",
"sub_question": "Show me total sales",
"measures": [{"name": "Sales","calculation": "Total","original_phrase": "total sales"}],
"dimensions": [],
"timegrain": null,
"timeframe": null,
"pattern": null,
"segments": [],
"breakdowns": [],
"unmatched_intents": []
}
]
}
```

Example 2: Complex (Planning + Discovery)
Q: "First calculate sales by territory, then compare with last year"
```json
{
"intent": "intents_discovery",
"discovery_results": [
{
"step_id": "step_1",
"sub_question": "Extract current year sales by territory",
"measures": [{"name": "Sales","calculation": "Total","original_phrase": "sales"}],
"dimensions": [{"name": "Territory","filter_value": null,"original_phrase": "territory"}],
"timegrain": {"phrase": "Daily"},
"timeframe": null,
"pattern": null,
"segments": [],
"breakdowns": [{"name":"Channel"},{"name":"Customer Type"},{"name":"Product Category"}],
"unmatched_intents": []
},
{
"step_id": "step_2",
"sub_question": "Extract last year sales by territory",
"measures": [{"name": "Sales","calculation": "Total","original_phrase": "sales"}],
"dimensions": [{"name": "Territory","filter_value": null,"original_phrase": "territory"}],
"timegrain": null,
"timeframe": {"phrase": "last year"},
"pattern": null,
"segments": [],
"breakdowns": [],
"unmatched_intents": []
},
{
"step_id": "step_3",
"sub_question": "Compare current vs last year performance",
"measures": [{"name": "Sales","calculation": "Comparison","original_phrase": "performance"}],
"dimensions": [{"name": "Territory","filter_value": null,"original_phrase": "territory"}],
"timegrain": null,
"timeframe": null,
"pattern": {"name":"Year-over-Year Comparison","phrase":"compare current vs last year"},
"segments": [],
"breakdowns": [],
"unmatched_intents": []
}
]
}
```

Example 3: Multiple Filter Values
Q: "Show me transactions for Desktop and Mobile channels"
```json
{
"intent": "intents_discovery",
"discovery_results": [
{
"step_id": "step_1",
"sub_question": "Show me transactions for Desktop and Mobile channels",
"measures": [{"name": "Transactions","calculation": "Total","original_phrase": "transactions"}],
"dimensions": [{"name": "Channel","filter_value": "Desktop, Mobile","original_phrase": "Desktop and Mobile channels"}],
"timegrain": null,
"timeframe": null,
"pattern": null,
"segments": [],
"breakdowns": [],
"unmatched_intents": []
}
]
}
```

Example 4: Complex Business Phrases
Q: "What is the daily average number of customers who complete their subscription renewals in Desktop?"
```json
{
"intent": "intents_discovery",
"discovery_results": [
{
"step_id": "step_1",
"sub_question": "What is the daily average number of customers who complete their subscription renewals in Desktop?",
"measures": [{"name":"Customers","calculation":"Average","original_phrase":"average number of customers"}],
"dimensions": [{"name":"Channel","filter_value":"Desktop","original_phrase":"Desktop"}],
"timegrain": {"phrase":"Daily"},
"timeframe": null,
"pattern": null,
"segments": [],
"breakdowns": [],
"unmatched_intents": [
{"phrase":"customers who complete their subscription renewals","type":"business_entity","reason":"Needs KB mapping to understand customer behavior"},
{"phrase":"complete","type":"business_action","reason":"Action that needs clarification on criteria"},
{"phrase":"subscription renewals","type":"business_entity","reason":"Subscription concept that needs KB mapping"}
]
}
]
}
```

Example 5: Ambiguity
Q: "Show me performance for ABC"
```json
{
"intent": "request_human_input",
"reasoning": "The term 'ABC' is ambiguous and could refer to multiple concepts",
"choices": ["Publisher: ABC Games","Region: ABC Territory","Product: ABC Suite","Don't know"],
"question": "What does 'ABC' refer to in your question?"
}
```"""
    
    return prompt_template

def load_trained_model(model_path: str):
    """Load the trained model for inference."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model
    base_model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 1024):
    """Generate response for a given question."""
    prompt_template = load_prompt_template()
    
    # Create input prompt
    input_text = prompt_template.format(question=question)
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.05,        # Much lower temperature for consistency
            do_sample=False,         # Deterministic generation
            repetition_penalty=1.2,  # Prevent infinite loops
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    response = generated_text[len(input_text):].strip()
    
    # Clean up the response - remove any leading/trailing whitespace and newlines
    response = response.strip()
    
    return response

def format_json_response(response: str) -> str:
    """Format and validate JSON response."""
    try:
        # Clean up the response - remove any leading/trailing whitespace
        cleaned_response = response.strip()
        
        # Try to parse as JSON
        parsed = json.loads(cleaned_response)
        
        # Pretty print the JSON
        formatted = json.dumps(parsed, indent=2)
        
        # Add validation status
        return f"✅ Valid JSON Response:\n{formatted}"
        
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded in text
        try:
            # Look for JSON-like content between curly braces
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                parsed = json.loads(json_content)
                formatted = json.dumps(parsed, indent=2)
                return f"✅ Valid JSON Response (extracted):\n{formatted}"
        except:
            pass
        
        return f"❌ Invalid JSON Response:\n{response}\n\nError: {str(e)}"

def interactive_chat():
    """Interactive chat interface."""
    print("🤖 Qwen BI Intent Discovery Chat")
    print("=" * 50)
    print("This model is trained to analyze BI questions and generate structured outputs.")
    print("Type 'quit' to exit, 'help' for examples.")
    print("=" * 50)
    
    # Load model
    model_path = "./qwen-bi-intent-model-second"
    
    try:
        model, tokenizer = load_trained_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model files are in the correct directory")
        return
    
    # Example questions
    examples = [
        "How many publishers have revenue above $1M?",
        "What is the average age of campaign managers?",
        "Show me the top 5 performing campaigns by engagement rate",
        "Compare revenue between Q1 and Q2 for all publishers",
        "Find customers who made purchases in the last 30 days",
        "What are the sales trends by region for this quarter?"
    ]
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\n📝 Example BI Questions:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                print("\n💡 Try asking questions about:")
                print("  • Publishers, campaigns, revenue")
                print("  • Customer data, sales, trends")
                print("  • Comparisons, rankings, filters")
                continue
            elif user_input.lower() == 'examples':
                print("\n🎯 Quick Examples:")
                for i, example in enumerate(examples, 1):
                    print(f"\n{i}. Question: {example}")
                    print("   Expected: JSON with measures, dimensions, filters")
                continue
            elif not user_input:
                continue
            
            # Generate response
            print("🤔 Thinking...")
            response = generate_response(model, tokenizer, user_input)
            
            # Format and display response
            print("\n🤖 Model Response:")
            print("-" * 30)
            
            # Check if response is empty or just whitespace
            if not response or response.strip() == "":
                print("❌ No response generated")
                print("💡 Try rephrasing your question")
                continue
                
            formatted_response = format_json_response(response)
            print(formatted_response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try rephrasing your question")

def test_model():
    """Test the model with predefined questions."""
    print("🧪 Testing Model with Sample Questions")
    print("=" * 50)
    
    model_path = "./qwen-bi-intent-model-initial"
    
    try:
        model, tokenizer = load_trained_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test questions
    test_questions = [
        "How many publishers have revenue above $1M?",
        "What is the average age of campaign managers?",
        "Show me the top 5 performing campaigns by engagement rate",
        "Compare revenue between Q1 and Q2 for all publishers"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        print("-" * 50)
        
        try:
            response = generate_response(model, tokenizer, question)
            formatted_response = format_json_response(response)
            print(formatted_response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        interactive_chat()
