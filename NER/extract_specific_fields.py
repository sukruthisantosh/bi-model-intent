#!/usr/bin/env python3
"""
Script to extract specific fields from balanced_questions_50_per_domain.json
Extracts: MEASURES, DIMENSIONS, TIMEFRAME, FILTERS, and question
"""

import json
import os

def extract_specific_fields(input_file, output_file):
    """
    Extract only the specified fields from the input JSON file
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract only the specified fields
    extracted_data = []
    for item in data:
        extracted_item = {
            "MEASURES": item.get("MEASURES", ""),
            "DIMENSIONS": item.get("DIMENSIONS", ""),
            "TIMEFRAME": item.get("TIMEFRAME", ""),
            "FILTERS": item.get("FILTERS", ""),
            "question": item.get("question", "")
        }
        extracted_data.append(extracted_item)
    
    # Write the extracted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(extracted_data)} records to {output_file}")
    print(f"Fields extracted: MEASURES, DIMENSIONS, TIMEFRAME, FILTERS, question")

if __name__ == "__main__":
    # File paths
    input_file = "data/processed/balanced_questions_50_per_domain.json"
    output_file = "data/processed/extracted_fields.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        exit(1)
    
    # Extract the fields
    extract_specific_fields(input_file, output_file)
    
    # Show a sample of the extracted data
    with open(output_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    print(f"\nSample of extracted data (first 3 records):")
    for i, item in enumerate(sample_data[:3]):
        print(f"\nRecord {i+1}:")
        for key, value in item.items():
            print(f"  {key}: {value}")
