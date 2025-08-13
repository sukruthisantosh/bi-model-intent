#!/usr/bin/env python3
"""
Convert Kaggle Spider Training Data to BI Domain
===============================================

This script converts 7,000 generic database queries to BI-related questions
while maintaining the complexity patterns and prompt logic structure.
"""

import json
import re
from typing import Dict, List, Any

# BI Domain Mapping - Generic BI terms, not schema-specific
BI_ENTITY_MAPPINGS = {
    # Generic BI entities
    "departments": ["publishers", "advertisers", "bidders", "campaigns"],
    "employees": ["users", "customers", "viewers", "clicks"],
    "budget": ["revenue", "spend", "cost", "budget"],
    "companies": ["publishers", "advertisers", "platforms", "networks"],
    "schools": ["websites", "apps", "channels", "sites"],
    "students": ["users", "visitors", "customers", "audience"],
    "customers": ["users", "viewers", "clients", "audience"],
    "products": ["ads", "campaigns", "creatives", "placements"],
    "sales": ["impressions", "clicks", "conversions", "revenue"],
    "orders": ["requests", "bids", "impressions", "clicks"],
    "payments": ["revenue", "spend", "cost", "payments"],
    "branches": ["locations", "regions", "markets", "territories"],
    "ships": ["campaigns", "ads", "creatives", "placements"],
    "films": ["ads", "videos", "creatives", "content"],
    "movies": ["ads", "videos", "creatives", "content"],
    "books": ["ads", "creatives", "content", "materials"],
    "sports": ["campaigns", "ads", "activities", "events"],
    "games": ["campaigns", "ads", "activities", "events"],
    "directors": ["managers", "leaders", "operators", "controllers"],
    
    # Generic BI measures
    "count": ["total", "count", "number", "sum"],
    "total": ["total", "sum", "count", "number"],
    "average": ["average", "mean", "avg", "typical"],
    "maximum": ["maximum", "highest", "peak", "top"],
    "minimum": ["minimum", "lowest", "bottom", "least"],
    "sum": ["sum", "total", "count", "number"],
    
    # Generic BI dimensions
    "name": ["name", "title", "label", "identifier"],
    "type": ["type", "category", "classification", "group"],
    "status": ["status", "state", "condition", "phase"],
    "location": ["location", "region", "area", "market"],
    "date": ["date", "time", "period", "duration"],
    "year": ["year", "period", "timeframe", "duration"],
    "month": ["month", "period", "timeframe", "duration"],
    "day": ["day", "date", "time", "period"],
    "age": ["age", "duration", "period", "time"],
    "rank": ["rank", "position", "order", "level"],
    "value": ["value", "amount", "quantity", "measure"],
    "amount": ["amount", "value", "quantity", "measure"],
    "price": ["price", "cost", "value", "amount"],
    "cost": ["cost", "price", "value", "amount"],
    "enrollment": ["engagement", "participation", "involvement", "activity"],
    "population": ["audience", "users", "viewers", "participants"],
    "market_value": ["revenue", "value", "worth", "performance"],
    "tonnage": ["volume", "capacity", "size", "amount"],
    "nationality": ["origin", "source", "location", "region"],
    "headquarters": ["location", "region", "market", "area"],
    "industry": ["category", "type", "sector", "domain"],
    "affiliation": ["association", "connection", "relationship", "partnership"],
    "foundation": ["creation", "start", "beginning", "launch"],
    "establishment": ["creation", "start", "beginning", "launch"],
    "creation": ["creation", "start", "beginning", "launch"],
    "birth": ["creation", "start", "beginning", "launch"],
    "born": ["created", "started", "launched", "established"],
    "acting": ["temporary", "interim", "provisional", "acting"],
    "temporary": ["temporary", "interim", "provisional", "acting"],
    "permanent": ["permanent", "fixed", "stable", "established"],
    "public": ["public", "open", "accessible", "available"],
    "private": ["private", "restricted", "exclusive", "limited"],
    "scholarship": ["premium", "preferred", "priority", "special"],
    "tryout": ["test", "trial", "evaluation", "assessment"],
    "decision": ["result", "outcome", "status", "state"],
    "position": ["role", "function", "position", "category"],
    "striker": ["primary", "main", "key", "important"],
    "goalkeeper": ["secondary", "support", "backup", "auxiliary"],
    "defender": ["secondary", "support", "backup", "auxiliary"],
    "midfielder": ["secondary", "support", "backup", "auxiliary"],
    "forward": ["primary", "main", "key", "important"],
    "coach": ["manager", "leader", "supervisor", "controller"],
    "team": ["group", "team", "unit", "collective"],
    "league": ["category", "group", "class", "division"],
    "season": ["period", "timeframe", "duration", "cycle"],
    "match": ["event", "activity", "session", "engagement"],
    "game": ["event", "activity", "session", "engagement"],
    "tournament": ["event", "activity", "session", "engagement"],
    "competition": ["event", "activity", "session", "engagement"],
    "championship": ["event", "activity", "session", "engagement"],
    "winner": ["winner", "leader", "top", "best"],
    "loser": ["loser", "bottom", "worst", "least"],
    "score": ["score", "rating", "performance", "result"],
    "goal": ["goal", "target", "objective", "aim"],
    "assist": ["assist", "support", "help", "aid"],
    "red_card": ["penalty", "violation", "issue", "problem"],
    "yellow_card": ["warning", "caution", "alert", "notice"],
    "foul": ["violation", "penalty", "issue", "problem"],
    "corner": ["corner", "edge", "boundary", "limit"],
    "penalty": ["penalty", "violation", "issue", "problem"],
    "substitution": ["change", "replacement", "switch", "update"],
    "injury": ["issue", "problem", "disruption", "interruption"],
    "suspension": ["suspension", "pause", "stop", "halt"],
    "transfer": ["transfer", "move", "change", "shift"],
    "contract": ["contract", "agreement", "deal", "arrangement"],
    "salary": ["salary", "payment", "compensation", "reward"],
    "bonus": ["bonus", "reward", "incentive", "benefit"],
    "commission": ["commission", "fee", "charge", "cost"],
    "tax": ["tax", "fee", "charge", "cost"],
    "insurance": ["insurance", "protection", "coverage", "safety"],
    "medical": ["medical", "health", "care", "treatment"],
    "physical": ["physical", "health", "fitness", "condition"],
    "mental": ["mental", "psychological", "emotional", "cognitive"],
    "technical": ["technical", "skill", "ability", "capability"],
    "tactical": ["tactical", "strategic", "planning", "approach"],
    "physical": ["physical", "health", "fitness", "condition"],
    "mental": ["mental", "psychological", "emotional", "cognitive"],
    "technical": ["technical", "skill", "ability", "capability"],
    "tactical": ["tactical", "strategic", "planning", "approach"],
}

# BI-specific question patterns
BI_QUESTION_PATTERNS = {
    "how_many": ["How many", "What is the total number of", "Count the number of", "Show the count of"],
    "list": ["List", "Show", "Display", "Provide"],
    "what_are": ["What are", "Which are", "Show all", "Display all"],
    "compare": ["Compare", "How does", "What is the difference between", "Show comparison of"],
    "top": ["Which are the top", "Show the top", "List the top", "What are the top"],
    "bottom": ["Which are the bottom", "Show the bottom", "List the bottom", "What are the bottom"],
    "average": ["What is the average", "Show the average", "Calculate the average", "Find the average"],
    "maximum": ["What is the maximum", "Show the maximum", "Find the maximum", "Calculate the maximum"],
    "minimum": ["What is the minimum", "Show the minimum", "Find the minimum", "Calculate the minimum"],
    "trend": ["What is the trend", "Show the trend", "How has", "What is the change"],
    "percentage": ["What percentage", "Show the percentage", "Calculate the percentage", "What is the share"],
}

def convert_question_to_bi(question: str) -> str:
    """Convert a generic question to BI-related question"""
    
    # Convert to lowercase for easier matching
    question_lower = question.lower()
    
    # Apply entity mappings
    converted_question = question
    for generic_term, bi_terms in BI_ENTITY_MAPPINGS.items():
        if generic_term in question_lower:
            # Choose a random BI term (for variety, we'll use the first one)
            bi_term = bi_terms[0]
            # Replace the term (case-insensitive)
            pattern = re.compile(re.escape(generic_term), re.IGNORECASE)
            converted_question = pattern.sub(bi_term, converted_question)
    
    # Add BI context if missing
    bi_terms_flat = [term for terms in BI_ENTITY_MAPPINGS.values() for term in terms]
    if not any(bi_term in converted_question.lower() for bi_term in bi_terms_flat):
        # Add generic BI context
        if "how many" in converted_question.lower():
            converted_question = converted_question.replace("how many", "How many users")
        elif "list" in converted_question.lower():
            converted_question = converted_question.replace("List", "List all campaigns")
        elif "what are" in converted_question.lower():
            converted_question = converted_question.replace("What are", "What are all the")
    
    return converted_question

def convert_output_to_bi(output: Dict[str, Any]) -> Dict[str, Any]:
    """Convert output to use BI terminology while maintaining structure"""
    
    converted_output = output.copy()
    
    # Convert discovery results
    if "discovery_results" in converted_output:
        for result in converted_output["discovery_results"]:
            # Convert measures
            if "measures" in result:
                for measure in result["measures"]:
                    if "name" in measure:
                        # Convert generic measure names to BI terms
                        if measure["name"] == "Count":
                            measure["name"] = "Total"
                        elif measure["name"] == "Average":
                            measure["name"] = "Average"
                        elif measure["name"] == "Max":
                            measure["name"] = "Maximum"
                        elif measure["name"] == "Min":
                            measure["name"] = "Minimum"
                        elif measure["name"] == "Sum":
                            measure["name"] = "Total"
            
            # Convert dimensions
            if "dimensions" in result:
                for dimension in result["dimensions"]:
                    if "name" in dimension:
                        # Convert generic dimension names to BI terms
                        if dimension["name"] == "Geographic":
                            dimension["name"] = "Region"
                        elif dimension["name"] == "Temporal":
                            dimension["name"] = "Time"
                        elif dimension["name"] == "Entity":
                            dimension["name"] = "Entity"
                        elif dimension["name"] == "Categorical":
                            dimension["name"] = "Category"
            
            # Convert unmatched intents to be more BI-relevant
            if "unmatched_intents" in result:
                for intent in result["unmatched_intents"]:
                    if "phrase" in intent:
                        # Convert generic phrases to BI terms
                        if intent["phrase"] in ["departments", "employees", "customers", "students"]:
                            intent["phrase"] = "users"
                        elif intent["phrase"] in ["budget", "sales", "revenue"]:
                            intent["phrase"] = "revenue"
                        elif intent["phrase"] in ["companies", "schools", "organizations"]:
                            intent["phrase"] = "publishers"
    
    return converted_output

def convert_training_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single training example to BI domain"""
    
    # Convert input question
    bi_question = convert_question_to_bi(example["input"])
    
    # Convert output
    bi_output = convert_output_to_bi(example["output"])
    
    return {
        "input": bi_question,
        "output": bi_output
    }

def main():
    """Convert the entire training dataset"""
    
    print("Loading training data...")
    with open("kaggle_spider_training_data.json", "r") as f:
        training_data = json.load(f)
    
    print(f"Converting {len(training_data)} examples to BI domain...")
    
    # Convert all examples
    bi_training_data = []
    for i, example in enumerate(training_data):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(training_data)} examples...")
        
        bi_example = convert_training_example(example)
        bi_training_data.append(bi_example)
    
    # Save converted data
    output_file = "bi_training_data.json"
    print(f"Saving converted data to {output_file}...")
    
    with open(output_file, "w") as f:
        json.dump(bi_training_data, f, indent=2)
    
    print(f"Conversion complete! Saved {len(bi_training_data)} BI examples.")
    
    # Print sample conversions
    print("\n=== Sample Conversions ===")
    for i in range(5):
        original = training_data[i]["input"]
        converted = bi_training_data[i]["input"]
        print(f"Original: {original}")
        print(f"Converted: {converted}")
        print("-" * 80)

if __name__ == "__main__":
    main()
