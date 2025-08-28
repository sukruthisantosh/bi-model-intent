#!/usr/bin/env python3
"""
Convert Complex Format to Simple Entity Recognition Format
========================================================

Convert the complex discovery_results format to the simple format used in training.
"""

import json

def convert_to_simple_format():
    """Convert complex format to simple entity recognition format."""
    
    # Load the complete training data (updated path after reorganization)
    with open('./legacy/training_data_fixed.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total examples in complete dataset: {len(all_data)}")
    
    # Convert ALL examples to simple format
    simple_data = []
    
    for i, example in enumerate(all_data):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(all_data)}")
        
        # Extract the question (it's called 'input' in this file)
        question = example.get('input', '')
        
        # Extract the discovery_results (it's nested under 'output')
        output = example.get('output', {})
        discovery_results = output.get('discovery_results', [])
        
        # Get the first discovery result (most examples have one)
        if discovery_results and len(discovery_results) > 0:
            first_result = discovery_results[0]
        else:
            first_result = {}
        
        # Extract dimensions, measures, etc. from the first result
        dimensions = []
        if first_result.get('dimensions'):
            for dim in first_result['dimensions']:
                if isinstance(dim, dict):
                    dimensions.append(dim.get('original_phrase', dim.get('name', '')))
                else:
                    dimensions.append(dim)
        
        measures = []
        if first_result.get('measures'):
            for measure in first_result['measures']:
                if isinstance(measure, dict):
                    measures.append(measure.get('original_phrase', measure.get('name', '')))
                else:
                    measures.append(measure)
        
        calculations = []
        if first_result.get('measures'):
            for measure in first_result['measures']:
                if isinstance(measure, dict):
                    calc = measure.get('calculation', '')
                    if calc:
                        calculations.append(calc)
        
        filters = []
        if first_result.get('dimensions'):
            for dim in first_result['dimensions']:
                if isinstance(dim, dict) and dim.get('filter_value'):
                    filters.append(dim.get('filter_value', ''))
        
        time_references = []
        if first_result.get('timeframe'):
            timeframe = first_result['timeframe']
            if isinstance(timeframe, dict):
                time_references.append(timeframe.get('phrase', ''))
            else:
                time_references.append(timeframe)
        
        # Convert to simple format
        simple_example = {
            'input': question,
            'output': {
                'dimensions': dimensions,
                'measures': measures,
                'calculations': calculations,
                'filters': filters,
                'time_references': time_references
            }
        }
        
        simple_data.append(simple_example)
    
    print(f"Converted {len(simple_data)} examples to simple format")
    
    # Save the complete dataset (updated path)
    complete_output_path = './data/training/current/training_data_complete_simple.json'
    with open(complete_output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved complete dataset to: {complete_output_path}")
    
    # Extract examples 200-300 for evaluation (updated path)
    eval_data = simple_data[200:300]
    eval_output_path = './data/training/current/eval_examples_200-300.json'
    
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
    
    # Verify that questions are not empty
    empty_questions = sum(1 for ex in simple_data if not ex['input'].strip())
    print(f"\n⚠️  Empty questions found: {empty_questions}")
    
    if empty_questions > 0:
        print("❌ WARNING: Some questions are empty! This will cause issues.")
        # Show first few non-empty examples
        non_empty = [ex for ex in simple_data if ex['input'].strip()]
        if non_empty:
            print(f"\n✅ First non-empty example:")
            print(json.dumps(non_empty[0], indent=2))
    else:
        print("✅ All questions have content!")
    
    return simple_data, eval_data

if __name__ == "__main__":
    convert_to_simple_format()
