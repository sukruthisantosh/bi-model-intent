import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_prompt_template(prompt_path):
    """Load the prompt template from file"""
    with open(prompt_path, "r") as f:
        return f.read()

def build_prompt(prompt_template, question):
    """Build the full prompt with the question"""
    return prompt_template.replace("{{ context.current_question }}", question)

def load_model_and_tokenizer():
    """Load Qwen model and tokenizer"""
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Use MPS (Apple Silicon GPU) if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading Qwen model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for GPU efficiency
        device_map=device
    ).eval()
    
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=1024):
    """Generate response from the Qwen model"""
    print("Generating response with Qwen...")
    
    # Format the prompt for Qwen (chat format)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for BI intent discovery."},
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize the input
    tokenized = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # Create attention mask (all tokens are valid)
    attention_mask = torch.ones_like(tokenized)
    
    # Move inputs to the same device as the model
    if device != "cpu":
        tokenized = tokenized.to(device)
        attention_mask = attention_mask.to(device)
    
    # Generate response with original parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][tokenized.shape[1]:], skip_special_tokens=True)
    return response.strip()

def save_result(question, response, output_path):
    """Save the question and response to a JSON file"""
    from datetime import datetime
    
    result = {
        "question": question,
        "response": response,
        "model": "Qwen2.5-7B-Instruct",
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved to: {output_path}")

def main():
    # Paths
    prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompts/prompt.txt")
    output_path = os.path.join(os.path.dirname(__file__), "../qwen_test_result.json")
    
    # Test question (simple one to start)
    test_question = "What are the ad impressions rendered and shown for last one month for all publishers for India?"
    
    print("=== Qwen Test with Qwen2.5-7B-Instruct ===")
    print(f"Question: {test_question}")
    print()
    
    try:
        # Load prompt template
        print("Loading prompt template...")
        prompt_template = load_prompt_template(prompt_path)
        
        # Build full prompt
        full_prompt = build_prompt(prompt_template, test_question)
        print(f"Prompt length: {len(full_prompt)} characters")
        print()
        
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer()
        
        # Generate response
        response = generate_response(model, tokenizer, full_prompt, device)
        
        # Save result
        save_result(test_question, response, output_path)
        
        # Print response
        print("\n=== Generated Response ===")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 