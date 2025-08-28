#!/usr/bin/env python3
"""
Model Evaluation Script
======================

Compare trained vs untrained models on entity recognition task.
Saves results to HuggingFace for sharing and reproducibility.
"""

import json
import time
import torch
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login, HfApi
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Caching is incompatible with gradient checkpointing")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations
MODEL_CONFIGS = {
    "trained": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": "ssuki/qwen-1.5b-entity-recognition",  # Your trained model
        "name": "Trained Qwen 1.5B"
    },
    "untrained": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": None,
        "name": "Untrained Qwen 1.5B"
    }
}

# Test data path
TEST_DATA_PATH = "./data/testing/test_examples_101-200.json"
PROMPT_PATH = "./data/prompts/entity_recognition_prompt.txt"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data():
    """Load test examples."""
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test examples")
    return data

def load_prompt_template():
    """Load the entity recognition prompt template."""
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(model_config: Dict[str, Any]):
    """Load model and tokenizer."""
    print(f"Loading {model_config['name']}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'], 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    # Load adapter if trained model
    if model_config['adapter_path']:
        try:
            model = PeftModel.from_pretrained(model, model_config['adapter_path'])
            print(f"Loaded adapter from {model_config['adapter_path']}")
        except Exception as e:
            print(f"Warning: Could not load adapter: {e}")
            print("Using base model instead")
    
    return model, tokenizer

# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, tokenizer, test_data: List[Dict], model_name: str):
    """Run inference on test data and measure performance."""
    print(f"\nRunning inference with {model_name}...")
    
    prompt_template = load_prompt_template()
    results = []
    
    # Disable gradient checkpointing for faster inference
    model.gradient_checkpointing_disable()
    
    for i, example in enumerate(test_data):
        print(f"Processing example {i+1}/{len(test_data)}", end="\r")
        
        # Prepare input
        input_text = prompt_template.format(question=example['input'])
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        inference_time = time.time() - start_time
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        # Try to parse JSON
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            parsed_response = {"error": "Invalid JSON", "raw_response": response}
        
        # Store result
        result = {
            "example_id": i + 1,
            "question": example['input'],
            "ground_truth": example['output'],
            "model_response": response,
            "parsed_response": parsed_response,
            "inference_time": inference_time,
            "model_name": model_name
        }
        
        results.append(result)
    
    print(f"\nCompleted inference with {model_name}")
    return results

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(results: List[Dict]):
    """Calculate evaluation metrics."""
    total_examples = len(results)
    valid_json_count = 0
    total_inference_time = 0
    
    # Count valid JSON responses
    for result in results:
        if "error" not in result['parsed_response']:
            valid_json_count += 1
        total_inference_time += result['inference_time']
    
    metrics = {
        "total_examples": total_examples,
        "valid_json_rate": valid_json_count / total_examples,
        "average_inference_time": total_inference_time / total_examples,
        "total_inference_time": total_inference_time
    }
    
    return metrics

def compare_entity_extraction(ground_truth: Dict, prediction: Dict):
    """Compare entity extraction accuracy."""
    if "error" in prediction:
        return {
            "dimensions_accuracy": 0.0,
            "measures_accuracy": 0.0,
            "calculations_accuracy": 0.0,
            "filters_accuracy": 0.0,
            "time_references_accuracy": 0.0,
            "overall_accuracy": 0.0
        }
    
    categories = ['dimensions', 'measures', 'calculations', 'filters', 'time_references']
    accuracies = {}
    
    for category in categories:
        gt_entities = set(ground_truth.get(category, []))
        pred_entities = set(prediction.get(category, []))
        
        if len(gt_entities) == 0 and len(pred_entities) == 0:
            accuracies[f"{category}_accuracy"] = 1.0
        elif len(gt_entities) == 0:
            accuracies[f"{category}_accuracy"] = 0.0
        else:
            # Calculate F1-like score
            intersection = len(gt_entities.intersection(pred_entities))
            precision = intersection / len(pred_entities) if len(pred_entities) > 0 else 0
            recall = intersection / len(gt_entities) if len(gt_entities) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracies[f"{category}_accuracy"] = f1
    
    # Overall accuracy (average of all categories)
    accuracies["overall_accuracy"] = sum(accuracies.values()) / len(accuracies)
    
    return accuracies

# ============================================================================
# RESULTS PROCESSING
# ============================================================================

def process_results(all_results: Dict[str, List[Dict]]):
    """Process and analyze all results."""
    print("\nProcessing results...")
    
    # Calculate metrics for each model
    metrics = {}
    detailed_results = []
    
    for model_name, results in all_results.items():
        # Basic metrics
        metrics[model_name] = calculate_metrics(results)
        
        # Detailed entity extraction analysis
        for result in results:
            entity_accuracy = compare_entity_extraction(
                result['ground_truth'], 
                result['parsed_response']
            )
            
            detailed_result = {
                "example_id": result['example_id'],
                "model_name": model_name,
                "question": result['question'],
                "ground_truth": result['ground_truth'],
                "model_response": result['model_response'],
                "inference_time": result['inference_time'],
                **entity_accuracy
            }
            
            detailed_results.append(detailed_result)
    
    return metrics, detailed_results

def create_summary_report(metrics: Dict, detailed_results: List[Dict]):
    """Create a summary report."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Total Examples: {model_metrics['total_examples']}")
        print(f"  Valid JSON Rate: {model_metrics['valid_json_rate']:.2%}")
        print(f"  Average Inference Time: {model_metrics['average_inference_time']:.3f}s")
        print(f"  Total Inference Time: {model_metrics['total_inference_time']:.2f}s")
    
    # Calculate average entity extraction accuracy
    df = pd.DataFrame(detailed_results)
    print(f"\nEntity Extraction Accuracy:")
    for model_name in metrics.keys():
        model_df = df[df['model_name'] == model_name]
        avg_overall = model_df['overall_accuracy'].mean()
        print(f"  {model_name}: {avg_overall:.2%}")

# ============================================================================
# SAVE TO HUGGINGFACE
# ============================================================================

def save_to_huggingface(metrics: Dict, detailed_results: List[Dict]):
    """Save results to HuggingFace Hub."""
    print("\nSaving results to HuggingFace Hub...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results
    results_data = {
        "evaluation_date": datetime.now().isoformat(),
        "test_data_path": TEST_DATA_PATH,
        "metrics": metrics,
        "detailed_results": detailed_results
    }
    
    # Save locally first
    results_file = f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved locally: {results_file}")
    
    # Create CSV for easy analysis
    df = pd.DataFrame(detailed_results)
    csv_file = f"evaluation_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV saved: {csv_file}")
    
    # Try to upload to HuggingFace
    try:
        api = HfApi()
        
        # Create dataset repository
        repo_name = f"ssuki/entity-recognition-evaluation-{timestamp}"
        
        # Upload files
        api.upload_file(
            path_or_fileobj=results_file,
            path_in_repo="evaluation_results.json",
            repo_id=repo_name,
            repo_type="dataset"
        )
        
        api.upload_file(
            path_or_fileobj=csv_file,
            path_in_repo="evaluation_results.csv",
            repo_id=repo_name,
            repo_type="dataset"
        )
        
        print(f"Results uploaded to: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"Warning: Could not upload to HuggingFace: {e}")
        print("Results saved locally only")
    
    return results_file, csv_file

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    print("Entity Recognition Model Evaluation")
    print("=" * 50)
    
    # Check HuggingFace login
    try:
        from huggingface_hub import whoami
        username = whoami()
        print(f"Logged in to HuggingFace as: {username}")
    except Exception:
        print("Warning: Not logged in to HuggingFace Hub")
        print("Results will be saved locally only")
    
    # Load test data
    test_data = load_test_data()
    
    # Run evaluation for each model
    all_results = {}
    
    for model_key, model_config in MODEL_CONFIGS.items():
        try:
            # Load model
            model, tokenizer = load_model_and_tokenizer(model_config)
            
            # Run inference
            results = run_inference(model, tokenizer, test_data, model_config['name'])
            all_results[model_key] = results
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {model_config['name']}: {e}")
            continue
    
    # Process results
    metrics, detailed_results = process_results(all_results)
    
    # Create summary
    create_summary_report(metrics, detailed_results)
    
    # Save results
    results_file, csv_file = save_to_huggingface(metrics, detailed_results)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved: {results_file}, {csv_file}")

if __name__ == "__main__":
    main()
