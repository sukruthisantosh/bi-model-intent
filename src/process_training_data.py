#!/usr/bin/env python3
"""
Process Training Data Through LLM
=================================

This script processes the training data through the LLM to fix all issues
and generate proper outputs according to the prompt.txt specifications.
"""

import os
import json
import openai
from typing import Dict, List, Any
import time
from tqdm import tqdm

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

class TrainingDataProcessor:
    def __init__(self):
        self.prompt_template = None
        self.context = ""
        self.cached_prompt = None
        self.processed_data = []
        
    def load_prompt_template(self, prompt_path: str):
        """Load and cache the prompt template"""
        print("Loading prompt template...")
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()
        print(f"Prompt template loaded ({len(self.prompt_template)} characters)")
        
    def load_context_files(self, model_dir: str):
        """Load context files (e.g., schema, model info)"""
        print("Loading context files...")
        if os.path.exists(model_dir):
            for fname in os.listdir(model_dir):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(".json"):
                    with open(fpath, "r") as f:
                        self.context += f"\n\n[{fname}]\n{f.read()}"
        print(f"Context loaded ({len(self.context)} characters)")
        
    def build_prompt(self, question: str) -> str:
        """Build the full prompt for a question"""
        if not self.prompt_template:
            raise ValueError("Prompt template not loaded")
            
        # Replace placeholders in the prompt template
        prompt = self.prompt_template.replace("{{ context.current_question }}", question)
        if "{{ context.model_files }}" in prompt:
            prompt = prompt.replace("{{ context.model_files }}", self.context)
            
        return prompt
        
    def process_question(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        """Process a single question through the LLM"""
        prompt = self.build_prompt(question)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",  # Using cheaper model for cost optimization
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for BI intent discovery. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent outputs
                    max_tokens=2048,  # Increased for complex responses
                    response_format={"type": "json_object"}  # Ensure JSON output
                )
                
                # Parse the response
                content = response.choices[0].message.content
                try:
                    result = json.loads(content)
                    return {
                        "input": question,
                        "output": result
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for question: {question[:50]}...")
                    print(f"Response: {content}")
                    if attempt == max_retries - 1:
                        # Return a basic structure if all retries fail
                        return {
                            "input": question,
                            "output": {
                                "intent": "intents_discovery",
                                "discovery_results": [{
                                    "step_id": "step_1",
                                    "sub_question": question,
                                    "measures": [],
                                    "dimensions": [],
                                    "timegrain": None,
                                    "timeframe": None,
                                    "pattern": None,
                                    "segments": [],
                                    "breakdowns": [],
                                    "unmatched_intents": []
                                }]
                            }
                        }
                    continue
                    
            except Exception as e:
                print(f"Error processing question (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def process_training_data(self, input_file: str, output_file: str, 
                            start_index: int = 0, end_index: int = None,
                            batch_size: int = 10):
        """Process the training data in batches"""
        print(f"Loading training data from {input_file}...")
        with open(input_file, "r") as f:
            data = json.load(f)
            
        total_examples = len(data)
        if end_index is None:
            end_index = total_examples
            
        data_to_process = data[start_index:end_index]
        print(f"Processing {len(data_to_process)} examples (indices {start_index}-{end_index-1})")
        
        # Process in batches
        for i in tqdm(range(0, len(data_to_process), batch_size), desc="Processing batches"):
            batch = data_to_process[i:i + batch_size]
            batch_results = []
            
            for j, example in enumerate(batch):
                question = example["input"]
                print(f"\nProcessing example {start_index + i + j + 1}: {question[:80]}...")
                
                try:
                    result = self.process_question(question)
                    batch_results.append(result)
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Failed to process example {start_index + i + j + 1}: {str(e)}")
                    # Add the original example as fallback
                    batch_results.append(example)
                    
            self.processed_data.extend(batch_results)
            
            # Save intermediate results every batch
            if (i + batch_size) % 50 == 0:
                self.save_intermediate_results(output_file, start_index, end_index)
                
        # Save final results
        self.save_final_results(output_file, start_index, end_index)
        
    def save_intermediate_results(self, output_file: str, start_index: int, end_index: int):
        """Save intermediate results"""
        temp_file = f"{output_file}.temp_{start_index}_{end_index}"
        with open(temp_file, "w") as f:
            json.dump(self.processed_data, f, indent=2)
        print(f"Intermediate results saved to {temp_file}")
        
    def save_final_results(self, output_file: str, start_index: int, end_index: int):
        """Save final results"""
        with open(output_file, "w") as f:
            json.dump(self.processed_data, f, indent=2)
        print(f"Final results saved to {output_file}")
        print(f"Processed {len(self.processed_data)} examples successfully")

def main():
    """Main function to process training data"""
    print("Training Data Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = TrainingDataProcessor()
    
    # Load prompt and context
    prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompts/prompt.txt")
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    
    processor.load_prompt_template(prompt_path)
    processor.load_context_files(model_dir)
    
    # Process training data
    input_file = os.path.join(os.path.dirname(__file__), "../training_data_fixed.json")
    output_file = os.path.join(os.path.dirname(__file__), "../training_data_llm_processed_1000.json")
    
    # You can adjust these parameters
    start_index = 0  # Start from the beginning
    end_index = 1000  # Process first 1000 examples
    batch_size = 5   # Process 5 examples at a time
    
    print(f"\nProcessing configuration:")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Start index: {start_index}")
    print(f"End index: {end_index}")
    print(f"Batch size: {batch_size}")
    
    # Confirm before starting
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Process the data
    processor.process_training_data(
        input_file=input_file,
        output_file=output_file,
        start_index=start_index,
        end_index=end_index,
        batch_size=batch_size
    )
    
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()
