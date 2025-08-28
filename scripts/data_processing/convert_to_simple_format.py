#!/usr/bin/env python3
"""
Convert Complex Format to Simple Entity Recognition Format
========================================================

Convert the complex discovery_results format to the simple format used in training.
"""

import json

def convert_to_simple_format():
    """Convert complex format to simple entity recognition format."""
    
    # Load the complete training data
    with open('./training_data_new/training_data_llm_processed_complete.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total examples in complete dataset: {len(all_data)}")
    
    # Convert ALL examples to simple format
    simple_data = []
    
    for i, example in enumerate(all_data):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(all_data)}")
        
        # Extract the question
        question = example.get('question', '')
        
        # Extract the discovery_results
        discovery_results = example.get('discovery_results', {})
        
        # Convert to simple format
        simple_example = {
            'input': question,
            'output': {
                'dimensions': discovery_results.get('dimensions', []),
                'measures': discovery_results.get('measures', []),
                'calculations': discovery_results.get('calculations', []),
                'filters': discovery_results.get('filters', []),
                'time_references': discovery_results.get('time_references', [])
            }
        }
        
        simple_data.append(simple_example)
    
    print(f"Converted {len(simple_data)} examples to simple format")
    
    # Save the complete dataset
    complete_output_path = './training_data_new/training_data_complete_simple.json'
    with open(complete_output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved complete dataset to: {complete_output_path}")
    
    # Extract examples 200-300 for evaluation
    eval_data = simple_data[200:300]
    eval_output_path = './training_data_new/eval_examples_200-300.json'
    
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved evaluation examples (200-300) to: {eval_output_path}")
    print(f"   Evaluation dataset size: {len(eval_data)} examples")
    
    # Show some statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {len(simple_data)}")
    print(f"   Training examples (0-199): {len(simple_data[:200])}")
    print(f"   Evaluation examples (200-299): {len(eval_data)}")
    print(f"   Remaining examples (300+): {len(simple_data[300:])}")
    
    # Show first example for verification
    print(f"\n🔍 First example format:")
    print(json.dumps(simple_data[0], indent=2))
    
    return simple_data, eval_data

if __name__ == "__main__":
    convert_to_simple_format()
