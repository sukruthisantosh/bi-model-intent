#!/usr/bin/env python3
"""
Simple runner script for the BI question generator
"""

import os
import sys

# Add the NER directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the generator
from scripts.generate_training_data import main

if __name__ == "__main__":
    print("=== GLiNER BI Training Data Generator ===")
    print("Using OPENAI_API_KEY from environment variables")
    print()
    
    # Check if API key is available in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please export your API key:")
        print("export OPENAI_API_KEY=your_api_key_here")
        print()
        sys.exit(1)
    
    # Run the generator
    exit_code = main()
    
    if exit_code == 0:
        print("\nGeneration completed successfully!")
        print("Check the NER/data/ directories for your generated files.")
    else:
        print("\nGeneration failed. Check the error messages above.")
    
    sys.exit(exit_code)
