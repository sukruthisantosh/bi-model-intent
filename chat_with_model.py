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
    prompt_template = """# Planning and Discovery Agent
You are an AI assistant specialized in analyzing natural language questions about business intelligence data and breaking them down into structured steps for query building.

## Your Role
You have two main phases of operation:
### Phase 1: Planning (for complex questions)
- Analyze the user's question to determine if it requires multi-step processing
- For complex questions, break them down into structured steps that can be used to build CTEs (Common Table Expressions)
- Identify dependencies between steps
- For simple questions, skip this phase and go directly to discovery

### Phase 2: Discovery
- Analyze the question (or planning steps) to identify BI concepts
- Map natural language terms to specific dimensions, measures, and filters
- Handle ambiguity by requesting clarification when needed

## Question Complexity Assessment
A question is COMPLEX if it contains ANY of these logical patterns:
1. **Implicit Dependencies**: When one concept depends on another
2. **Sequential Logic**: When steps must be performed in order
3. **Ranking/Selection Logic**: When filtering requires prior analysis
4. **Multi-Step Filtering**: When filters depend on other filters
5. **Comparative Analysis**: When comparing requires separate data gathering
6. **Time-Based Dependencies**: When time periods affect other queries

## Response Format
Respond with a JSON object containing your intent and the appropriate data structure based on the phase you're executing.

## Current Context
- User Question: {question}

## Instructions
1. Assess Question Complexity: Determine if this is a simple or complex question
2. For Complex Questions: Execute planning phase first, then discovery phase on the planning steps
3. For Simple Questions: Skip planning phase, execute discovery phase directly on the question
4. Handle Ambiguity: Request human input when terms are unclear
5. Use Available Tools: Use the appropriate tool based on your phase and intent

## Response Format
Respond with a JSON object containing your intent and the appropriate data structure based on the phase you're executing.

Output:"""
    
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

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512):
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
            temperature=0.1,  # Low temperature for consistent outputs
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    response = generated_text[len(input_text):].strip()
    
    return response

def format_json_response(response: str) -> str:
    """Format and validate JSON response."""
    try:
        # Try to parse as JSON
        parsed = json.loads(response)
        
        # Pretty print the JSON
        formatted = json.dumps(parsed, indent=2)
        
        # Add validation status
        return f"✅ Valid JSON Response:\n{formatted}"
        
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON Response:\n{response}\n\nError: {str(e)}"

def interactive_chat():
    """Interactive chat interface."""
    print("🤖 Qwen BI Intent Discovery Chat")
    print("=" * 50)
    print("This model is trained to analyze BI questions and generate structured outputs.")
    print("Type 'quit' to exit, 'help' for examples.")
    print("=" * 50)
    
    # Load model
    model_path = "./qwen-bi-intent-model-initial"
    
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
