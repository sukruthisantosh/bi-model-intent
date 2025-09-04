#!/usr/bin/env python3
"""
Question Annotation Script using NER
Annotates a small sample of questions to extract entities like MEASURES, DIMENSIONS, TIMEFRAME, and FILTERS
"""

import json
import re
from typing import List, Dict, Any
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_questions(file_path: str, num_questions: int = 10) -> List[Dict[str, Any]]:
    """Load a sample of questions from the extracted fields file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Take first N questions
        return data[:num_questions]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def extract_measures(question: str) -> List[str]:
    """Extract MEASURES from the question using pattern matching"""
    measures = []
    
    # Common measure patterns
    measure_patterns = [
        r'\b(revenue|sales|profit|cost|margin|discount|returns?|orders?|items?|products?|customers?|users?|leads?|impressions?|clicks?|spend|acquisition|satisfaction|complaints?|support|tickets?|shipments?|deliveries?|inventory|production|suppliers?|properties?|rentals?|subscriptions?|churn|retention|engagement|conversion|roi|ctr|cpa|cpc|cpm)\b',
        r'\b(rate|percentage|ratio|average|total|count|sum|amount|value|volume|number|quantity|time|duration|speed|efficiency|performance|growth|change|trend)\b',
        r'\b(active|new|returning|loyal|premium|basic|standard|deluxe|express|regular|seasonal|promotional|corporate|wholesale|retail|online|mobile|b2b|b2c)\b'
    ]
    
    question_lower = question.lower()
    
    for pattern in measure_patterns:
        matches = re.findall(pattern, question_lower)
        measures.extend(matches)
    
    # Remove duplicates and clean up
    measures = list(set(measures))
    measures = [m.capitalize() for m in measures if len(m) > 2]
    
    return measures if measures else ["none"]

def extract_dimensions(question: str) -> List[str]:
    """Extract DIMENSIONS from the question using pattern matching"""
    dimensions = []
    
    # Common dimension patterns
    dimension_patterns = [
        r'\b(country|region|state|city|area|location|zone|territory|market|geography)\b',
        r'\b(product|category|brand|sku|item|type|model|variant|line|family|group|class)\b',
        r'\b(customer|user|segment|tier|level|status|type|group|cohort|demographic|age|gender|income)\b',
        r'\b(channel|source|medium|platform|device|browser|os|app|website|store|outlet|branch)\b',
        r'\b(campaign|ad|creative|placement|keyword|audience|target|segment|strategy|tactic)\b',
        r'\b(month|quarter|year|week|day|hour|period|season|fiscal|calendar|date|time)\b',
        r'\b(department|team|employee|role|position|function|division|unit|section|group)\b',
        r'\b(supplier|vendor|partner|contractor|manufacturer|distributor|retailer|wholesaler)\b'
    ]
    
    question_lower = question.lower()
    
    for pattern in dimension_patterns:
        matches = re.findall(pattern, question_lower)
        dimensions.extend(matches)
    
    # Remove duplicates and clean up
    dimensions = list(set(dimensions))
    dimensions = [d.capitalize() for d in dimensions if len(d) > 2]
    
    return dimensions if dimensions else ["none"]

def extract_timeframe(question: str) -> str:
    """Extract TIMEFRAME from the question using pattern matching"""
    timeframe_patterns = [
        r'\b(last|past|previous|prior|recent|current|this|next|upcoming|following)\s+(day|week|month|quarter|year|decade|century)\b',
        r'\b(Q[1-4]\s+\d{4})\b',  # Q1 2024
        r'\b(\d{4})\b',  # 2024
        r'\b(ytd|year\s*to\s*date|mtd|month\s*to\s*date|qtd|quarter\s*to\s*date)\b',
        r'\b(rolling|moving|sliding)\s+\d+\s+(day|week|month|quarter|year)s?\b',
        r'\b(black\s*friday|christmas|holiday|season|summer|winter|spring|fall|autumn)\s+(week|month|season)?\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
        r'\b(yesterday|today|tomorrow|now|currently|recently|lately)\b'
    ]
    
    question_lower = question.lower()
    
    for pattern in timeframe_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return match.group(0).title()
    
    return "none"

def extract_filters(question: str) -> List[str]:
    """Extract FILTERS from the question using pattern matching"""
    filters = []
    
    # Common filter patterns
    filter_patterns = [
        r'\b(excluding|excluding|except|not|without|excluding|excluding)\s+(\w+)\b',
        r'\b(only|just|specifically|particularly|especially|namely)\s+(\w+)\b',
        r'\b(top|bottom|best|worst|highest|lowest|maximum|minimum)\s+\d+\b',
        r'\b(above|below|over|under|more\s+than|less\s+than|greater\s+than|less\s+than)\s+[\$\d]+\b',
        r'\b(active|inactive|enabled|disabled|on|off|yes|no|true|false)\b',
        r'\b(new|returning|loyal|premium|basic|standard|deluxe|express|regular)\b',
        r'\b(seasonal|promotional|corporate|wholesale|retail|online|mobile|b2b|b2c)\b',
        r'\b(clearance|sale|discount|promotion|offer|deal|bundle|package|set)\b'
    ]
    
    question_lower = question.lower()
    
    for pattern in filter_patterns:
        matches = re.findall(pattern, question_lower)
        if matches:
            if isinstance(matches[0], tuple):
                filters.extend([' '.join(match) for match in matches])
            else:
                filters.extend(matches)
    
    # Remove duplicates and clean up
    filters = list(set(filters))
    filters = [f.capitalize() for f in filters if len(f) > 2]
    
    return filters if filters else ["none"]

def annotate_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate a single question with extracted entities"""
    question_text = question_data.get('question', '')
    
    # Clean the question text (remove quotes)
    question_text = question_text.strip('"')
    
    # Extract entities
    measures = extract_measures(question_text)
    dimensions = extract_dimensions(question_text)
    timeframe = extract_timeframe(question_text)
    filters = extract_filters(question_text)
    
    # Create annotation result
    annotation = {
        'original_question': question_text,
        'extracted_measures': measures,
        'extracted_dimensions': dimensions,
        'extracted_timeframe': timeframe,
        'extracted_filters': filters,
        'ground_truth': {
            'MEASURES': question_data.get('MEASURES', 'none'),
            'DIMENSIONS': question_data.get('DIMENSIONS', 'none'),
            'TIMEFRAME': question_data.get('TIMEFRAME', 'none'),
            'FILTERS': question_data.get('FILTERS', 'none')
        }
    }
    
    return annotation

def main():
    """Main function to annotate questions"""
    # File paths
    input_file = os.path.join('data', 'processed', 'extracted_fields.json')
    output_file = os.path.join('data', 'results', 'ner_annotations_10_questions.json')
    
    print("Loading questions for annotation...")
    questions = load_questions(input_file, num_questions=10)
    
    if not questions:
        print("No questions loaded. Exiting.")
        return
    
    print(f"Loaded {len(questions)} questions for annotation.")
    print("\n" + "="*80)
    
    # Annotate each question
    annotations = []
    for i, question_data in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Original: {question_data.get('question', '')}")
        
        annotation = annotate_question(question_data)
        annotations.append(annotation)
        
        # Display extraction results
        print(f"Extracted MEASURES: {annotation['extracted_measures']}")
        print(f"Extracted DIMENSIONS: {annotation['extracted_dimensions']}")
        print(f"Extracted TIMEFRAME: {annotation['extracted_timeframe']}")
        print(f"Extracted FILTERS: {annotation['extracted_filters']}")
        
        # Display ground truth for comparison
        print(f"Ground Truth MEASURES: {annotation['ground_truth']['MEASURES']}")
        print(f"Ground Truth DIMENSIONS: {annotation['ground_truth']['DIMENSIONS']}")
        print(f"Ground Truth TIMEFRAME: {annotation['ground_truth']['TIMEFRAME']}")
        print(f"Ground Truth FILTERS: {annotation['ground_truth']['FILTERS']}")
        
        print("-" * 60)
    
    # Save annotations
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnnotations saved to: {output_file}")
        
        # Calculate accuracy metrics
        calculate_accuracy(annotations)
        
    except Exception as e:
        print(f"Error saving annotations: {e}")

def calculate_accuracy(annotations: List[Dict[str, Any]]):
    """Calculate accuracy metrics for the annotations"""
    total_questions = len(annotations)
    correct_measures = 0
    correct_dimensions = 0
    correct_timeframes = 0
    correct_filters = 0
    
    for annotation in annotations:
        # Check MEASURES accuracy
        extracted = set(annotation['extracted_measures'])
        ground_truth = set([annotation['ground_truth']['MEASURES'].lower()] if annotation['ground_truth']['MEASURES'] != 'none' else [])
        
        if extracted == ground_truth or (not extracted and not ground_truth):
            correct_measures += 1
        
        # Check DIMENSIONS accuracy
        extracted = set(annotation['extracted_dimensions'])
        ground_truth = set([annotation['ground_truth']['DIMENSIONS'].lower()] if annotation['ground_truth']['DIMENSIONS'] != 'none' else [])
        
        if extracted == ground_truth or (not extracted and not ground_truth):
            correct_dimensions += 1
        
        # Check TIMEFRAME accuracy
        extracted = annotation['extracted_timeframe'].lower()
        ground_truth = annotation['ground_truth']['TIMEFRAME'].lower()
        
        if extracted == ground_truth or (extracted == 'none' and ground_truth == 'none'):
            correct_timeframes += 1
        
        # Check FILTERS accuracy
        extracted = set(annotation['extracted_filters'])
        ground_truth = set([annotation['ground_truth']['FILTERS'].lower()] if annotation['ground_truth']['FILTERS'] != 'none' else [])
        
        if extracted == ground_truth or (not extracted and not ground_truth):
            correct_filters += 1
    
    print(f"\n" + "="*80)
    print("ACCURACY METRICS:")
    print(f"Total Questions: {total_questions}")
    print(f"MEASURES Accuracy: {correct_measures}/{total_questions} ({correct_measures/total_questions*100:.1f}%)")
    print(f"DIMENSIONS Accuracy: {correct_dimensions}/{total_questions} ({correct_dimensions/total_questions*100:.1f}%)")
    print(f"TIMEFRAME Accuracy: {correct_timeframes}/{total_questions} ({correct_timeframes/total_questions*100:.1f}%)")
    print(f"FILTERS Accuracy: {correct_filters}/{total_questions} ({correct_filters/total_questions*100:.1f}%)")
    
    overall_accuracy = (correct_measures + correct_dimensions + correct_timeframes + correct_filters) / (total_questions * 4)
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")

if __name__ == "__main__":
    main()
