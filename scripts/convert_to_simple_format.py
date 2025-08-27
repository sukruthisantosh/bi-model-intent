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
    
    # Extract examples 101-200 (indices 100-199)
    complex_data = all_data[100:200]
    
    print(f"Converting {len(complex_data)} examples (indices 100-199)")
    
    # Convert to simple format
    simple_data = []
    
    for i, example in enumerate(complex_data):
        discovery_result = example['output']['discovery_results'][0]  # Take first step
        
        # Extract dimensions
        dimensions = []
        if discovery_result.get('dimensions'):
            for dim in discovery_result['dimensions']:
                if isinstance(dim, dict):
                    dimensions.append(dim.get('original_phrase', dim.get('name', '')))
                else:
                    dimensions.append(dim)
        
        # Extract measures
        measures = []
        if discovery_result.get('measures'):
            for measure in discovery_result['measures']:
                if isinstance(measure, dict):
                    measures.append(measure.get('original_phrase', measure.get('name', '')))
                else:
                    measures.append(measure)
        
        # Extract calculations (from measures or patterns)
        calculations = []
        if discovery_result.get('measures'):
            for measure in discovery_result['measures']:
                if isinstance(measure, dict):
                    calc = measure.get('calculation', '')
                    if calc:
                        calculations.append(calc)
        
        # Extract filters (from dimensions with filter_value)
        filters = []
        if discovery_result.get('dimensions'):
            for dim in discovery_result['dimensions']:
                if isinstance(dim, dict) and dim.get('filter_value'):
                    filters.append(dim.get('filter_value', ''))
        
        # Extract time references
        time_references = []
        if discovery_result.get('timeframe'):
            timeframe = discovery_result['timeframe']
            if isinstance(timeframe, dict):
                time_references.append(timeframe.get('phrase', ''))
            else:
                time_references.append(timeframe)
        
        # Create simple format example
        simple_example = {
            "input": example['input'],
            "output": {
                "dimensions": dimensions,
                "measures": measures,
                "calculations": calculations,
                "filters": filters,
                "time_references": time_references
            }
        }
        
        simple_data.append(simple_example)
    
    # Show a few examples
    print("\nSample converted examples:")
    for i, example in enumerate(simple_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['input']}")
        print(f"  Output: {example['output']}")
    
    # Save test set in simple format
    with open('./test_examples_101-200.json', 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTest set saved to: ./test_examples_simple.json")
    
    return simple_data

if __name__ == "__main__":
    convert_to_simple_format()
