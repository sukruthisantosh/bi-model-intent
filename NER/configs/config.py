"""Configuration for GLiNER training data generation"""

import os

# OpenAI Configuration - gets from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini" 

# Generation Parameters
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 16000  # Adjust based on model limits

# Data Generation Settings
TOTAL_QUESTIONS = 500
DOMAINS = ["ecommerce", "retail", "ads", "finance", "saas", "logistics", "manufacturing", "supply_chain", "real_estate", "telecommunications"]
QUESTIONS_PER_DOMAIN = 50
BATCH_SIZE = 10  # Generate in smaller batches to avoid issues

# Output Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TRAINING_DATA_DIR = "data/training"

# GLiNER Entity Types for BI Questions
ENTITY_TYPES = [
    "MEASURE",      # Revenue, Orders, Users, etc.
    "DIMENSION",    # Country, Product, Campaign, etc.
    "CALCULATION",  # sum, avg, count, etc.
    "TIMEFRAME",    # last quarter, 2024-01-01 to 2024-12-31, etc.
    "TIMEGRAIN",    # day, week, month, etc.
    "FILTER",       # Country = UK, Exclude = returns, etc.
    "DOMAIN",       # ecommerce, retail, etc.
    "COMPLEXITY"    # simple, complex, ambiguous
]
