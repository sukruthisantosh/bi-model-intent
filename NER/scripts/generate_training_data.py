"""
GLiNER Training Data Generator for BI Questions
Generates training data using OpenAI API with structured BI question prompts
"""

import os
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any
import openai
from openai import OpenAI

# Add parent directory to path to import config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *

class BIQuestionGenerator:
    def __init__(self):
        """Initialize the generator with OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your environment or .env file")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.generated_questions = []
        
        # Ensure output directories exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    def get_generation_prompt(self) -> str:
        """Return the complete prompt for BI question generation"""
        return """You are an expert Business-Intelligence question generator + extractor. 
Your job is to produce exactly 500 unique natural-language BI questions across 10 domains, and for each question immediately output a concise inline extraction of the BI pieces: MEASURES, DIMENSIONS, CALCULATION, TIMEFRAME, TIMEGRAIN (plus FILTERS and NOTES when applicable). The goal is to cover a wide range of real-world phrasing, paraphrases, complexity levels, time expressions, and calculation types.

RULES (must follow exactly):
1. Produce exactly 500 items, one item per line. Do not output anything else (no preface, no summary, no final line).
2. Use this exact single-line format for each item (fields separated by " || "):
   Q: <natural-language question> || DOMAIN: <domain> || COMPLEXITY: <simple|complex|ambiguous> || MEASURES: <comma-separated measures> || DIMENSIONS: <comma-separated dimensions> || CALCULATION: <sum|avg|count|distinct_count|median|rate|percent_change|cagr|ratio|moving_average|percent_of_total|difference|comparison|custom_formula> || TIMEFRAME: <phrase, e.g. "last 30 days", "Q2 2024", "2023-01-01 to 2023-12-31", "YTD", "past 12 months", "rolling 7 days"> || TIMEGRAIN: <day|week|month|quarter|year|hour|none|custom> || FILTERS: <comma-separated filters or "none"> || NOTES: <short clarifier or "none">
3. Domains to use (exact strings): ecommerce, retail, ads, finance, saas, logistics, manufacturing, supply_chain, real_estate, telecommunications
   - Produce exactly 50 questions for each domain (50 * 10 = 500).
4. Complexity distribution per domain: roughly 55% simple, 30% complex, 15% ambiguous/edge (approx; you must still produce exactly 50 per domain).
   - Simple: single measure + straightforward filters or none.
   - Complex: multi-step logic, ranking/selection, compare vs historical, nested filters, multiple measures/aggregations or derived metrics.
   - Ambiguous/edge: intentionally ambiguous entity names, ambiguous phrasing, or missing context (label as ambiguous so human grounding/KB mapping would be required).
5. Coverage requirements (try to satisfy all):
   - Timeframes: include relative (last 7 days, last month), named periods (Q1 2024), absolute ranges, YTD, rolling windows.
   - Timegrains: include day/week/month/quarter/year and some "none" (when timeframe not applicable).
   - Calculations: include a broad set from the CALCULATION list above; include ratio/percent_of_total/moving average/percent_change examples.
   - Filters: include equality, range, multiple values (e.g., "country = UK, US"), negative filters ("excluding returns"), and segments (cohorts).
   - Ranking/Top-K: include top N requests (e.g., "top 10 products by revenue").
   - Comparisons: include YoY/WoW/diff vs previous period.
6. Variation & paraphrase: For similar intents, vary phrasing (e.g., "How much", "What was", "Show me", "Give me", "List the top...").
7. Uniqueness: Avoid near-duplicates. Each Q should bring a new combination of domain, measures, dimensions, timeframe, or phrasing.
8. Annotations: MEASURES and DIMENSIONS should be short canonical phrases (e.g., "Revenue", "Orders", "Active Users", "Impressions", "Ad Spend", "Gross Margin", "Churn Rate"). If a question contains multiple measures/dimensions, list them comma-separated.
9. If a question is complex and naturally splits into steps (e.g., "first X then Y"), still output a single Q line but set COMPLEXITY to "complex" and list all measures/dimensions referenced.
10. For ambiguous phrases, set COMPLEXITY to "ambiguous" and include a concise NOTE explaining the ambiguity (e.g., "ABC could be publisher or product").
11. Strictly follow the line format and field names. Do not output JSON or other structures—use the exact format above.
12. Generate content neutrally; do not include real personal data or protected health details. Healthcare questions should be operational (counts, appointment rates), not patient-identifying.

EXAMPLES (follow this style exactly):

Q: "Show me total revenue for last quarter by country" || DOMAIN: ecommerce || COMPLEXITY: simple || MEASURES: Revenue || DIMENSIONS: Country || CALCULATION: sum || TIMEFRAME: last quarter || TIMEGRAIN: quarter || FILTERS: none || NOTES: none

Q: "What are the top 10 SKUs by revenue in the UK for Black Friday week, excluding returns?" || DOMAIN: retail || COMPLEXITY: complex || MEASURES: Revenue || DIMENSIONS: SKU || CALCULATION: sum || TIMEFRAME: Black Friday week 2024 || TIMEGRAIN: day || FILTERS: Country = UK, Exclude = returns || NOTES: top-k ranking

Q: "How many active paying users did we have in January?" || DOMAIN: saas || COMPLEXITY: simple || MEASURES: Active Paying Users || DIMENSIONS: none || CALCULATION: count || TIMEFRAME: 2024-01-01 to 2024-01-31 || TIMEGRAIN: month || FILTERS: none || NOTES: none

Q: "Compare ad spend vs. revenue by campaign for the past 3 months and show percent change month over month" || DOMAIN: ads || COMPLEXITY: complex || MEASURES: Ad Spend, Revenue || DIMENSIONS: Campaign || CALCULATION: percent_change || TIMEFRAME: past 3 months || TIMEGRAIN: month || FILTERS: none || NOTES: include MoM percent change

Q: "Performance for ABC" || DOMAIN: media || COMPLEXITY: ambiguous || MEASURES: none || DIMENSIONS: none || CALCULATION: none || TIMEFRAME: none || TIMEGRAIN: none || FILTERS: none || NOTES: "ABC" ambiguous - could be channel, show, or advertiser

Now generate the 500 lines exactly following the format and rules above."""

    def generate_batch(self, batch_num: int, questions_per_batch: int = BATCH_SIZE) -> List[str]:
        """Generate a batch of questions using OpenAI API"""
        print(f"Generating batch {batch_num + 1} with {questions_per_batch} questions...")
        print(f"DEBUG: API Key present: {'Yes' if OPENAI_API_KEY else 'No'}")
        print(f"DEBUG: API Key starts with: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...")
        
        # Modify prompt for batch generation
        batch_prompt = self.get_generation_prompt().replace(
            "exactly 500 unique natural-language BI questions",
            f"exactly {questions_per_batch} unique natural-language BI questions"
        ).replace(
            "Produce exactly 500 items",
            f"Produce exactly {questions_per_batch} items"
        ).replace(
            "50 * 10 = 500",
            f"distribute evenly across domains"
        )
        
        try:
            print(f"DEBUG: Making actual API call to OpenAI...")
            print(f"DEBUG: Model: {OPENAI_MODEL}")
            print(f"DEBUG: Max tokens: {MAX_TOKENS}")
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert BI question generator. Follow the format exactly."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS
            )
            
            print(f"DEBUG: API call successful!")
            print(f"DEBUG: Response ID: {response.id}")
            print(f"DEBUG: Usage: {response.usage}")
            
            content = response.choices[0].message.content.strip()
            questions = [line.strip() for line in content.split('\n') if line.strip()]
            
            print(f"Generated {len(questions)} questions in batch {batch_num + 1}")
            print(f"DEBUG: First few lines of response:")
            for i, line in enumerate(questions[:3]):
                print(f"  {i+1}: {line[:100]}...")
            
            return questions
            
        except Exception as e:
            print(f"ERROR generating batch {batch_num + 1}: {e}")
            print(f"ERROR type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return []
    
    def parse_question_line(self, line: str) -> Dict[str, Any]:
        """Parse a single question line into structured format"""
        try:
            # Split by " || " delimiter
            parts = line.split(" || ")
            if len(parts) < 8:
                return None
            
            parsed = {}
            for part in parts:
                if ": " in part:
                    key, value = part.split(": ", 1)
                    parsed[key.strip()] = value.strip()
            
            # Extract the question text (remove "Q: " prefix)
            if "Q" in parsed:
                parsed["question"] = parsed["Q"]
                del parsed["Q"]
            
            return parsed
            
        except Exception as e:
            print(f"Error parsing line: {line[:100]}... - {e}")
            return None
    
    def convert_to_gliner_format(self, parsed_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert parsed questions to GLiNER training format"""
        gliner_data = []
        
        for q_data in parsed_questions:
            if not q_data or "question" not in q_data:
                continue
            
            question = q_data["question"]
            
            # Create entity annotations based on the parsed data
            entities = []
            
            # Add entities for each field
            for field_name, entity_type in [
                ("MEASURES", "MEASURE"),
                ("DIMENSIONS", "DIMENSION"), 
                ("CALCULATION", "CALCULATION"),
                ("TIMEFRAME", "TIMEFRAME"),
                ("TIMEGRAIN", "TIMEGRAIN"),
                ("FILTERS", "FILTER"),
                ("DOMAIN", "DOMAIN"),
                ("COMPLEXITY", "COMPLEXITY")
            ]:
                if field_name in q_data and q_data[field_name] not in ["none", ""]:
                    # For comma-separated values, create separate entities
                    values = [v.strip() for v in q_data[field_name].split(",")]
                    for value in values:
                        if value and value != "none":
                            entities.append({
                                "label": entity_type,
                                "text": value,
                                "field": field_name
                            })
            
            gliner_item = {
                "text": question,
                "entities": entities,
                "metadata": q_data
            }
            
            gliner_data.append(gliner_item)
        
        return gliner_data
    
    def generate_all_questions(self) -> List[Dict[str, Any]]:
        """Generate all questions in batches"""
        all_questions = []
        num_batches = (TOTAL_QUESTIONS + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_num in range(num_batches):
            questions_in_batch = min(BATCH_SIZE, TOTAL_QUESTIONS - len(all_questions))
            
            batch_questions = self.generate_batch(batch_num, questions_in_batch)
            
            # Parse each question
            for line in batch_questions:
                parsed = self.parse_question_line(line)
                if parsed:
                    all_questions.append(parsed)
            
            # Rate limiting
            if batch_num < num_batches - 1:
                time.sleep(2)
        
        return all_questions
    
    def save_data(self, raw_questions: List[Dict[str, Any]], gliner_data: List[Dict[str, Any]]):
        """Save the generated data in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw questions
        raw_file = os.path.join(RAW_DATA_DIR, f"bi_questions_raw_{timestamp}.json")
        with open(raw_file, 'w') as f:
            json.dump(raw_questions, f, indent=2)
        print(f"Saved {len(raw_questions)} raw questions to {raw_file}")
        
        # Save GLiNER format
        gliner_file = os.path.join(TRAINING_DATA_DIR, f"gliner_training_data_{timestamp}.json")
        with open(gliner_file, 'w') as f:
            json.dump(gliner_data, f, indent=2)
        print(f"Saved {len(gliner_data)} GLiNER training examples to {gliner_file}")
        
        # Save statistics
        stats = self.generate_statistics(raw_questions)
        stats_file = os.path.join(PROCESSED_DATA_DIR, f"generation_stats_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_file}")
        
        return raw_file, gliner_file, stats_file
    
    def generate_statistics(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the generated questions"""
        stats = {
            "total_questions": len(questions),
            "domains": {},
            "complexity": {},
            "calculations": {},
            "timegrains": {},
            "generation_timestamp": datetime.now().isoformat()
        }
        
        for q in questions:
            # Domain distribution
            domain = q.get("DOMAIN", "unknown")
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            
            # Complexity distribution
            complexity = q.get("COMPLEXITY", "unknown")
            stats["complexity"][complexity] = stats["complexity"].get(complexity, 0) + 1
            
            # Calculation types
            calc = q.get("CALCULATION", "unknown")
            stats["calculations"][calc] = stats["calculations"].get(calc, 0) + 1
            
            # Time grains
            grain = q.get("TIMEGRAIN", "unknown")
            stats["timegrains"][grain] = stats["timegrains"].get(grain, 0) + 1
        
        return stats
    
    def run(self):
        """Main execution method"""
        print("Starting BI Question Generation for GLiNER Training...")
        print(f"Target: {TOTAL_QUESTIONS} questions across {len(DOMAINS)} domains")
        print(f"Using model: {OPENAI_MODEL}")
        
        # Generate questions
        raw_questions = self.generate_all_questions()
        print(f"Generated {len(raw_questions)} questions total")
        
        # Convert to GLiNER format
        gliner_data = self.convert_to_gliner_format(raw_questions)
        print(f"Converted to {len(gliner_data)} GLiNER training examples")
        
        # Save all data
        raw_file, gliner_file, stats_file = self.save_data(raw_questions, gliner_data)
        
        print("\nGeneration complete!")
        print(f"Raw data: {raw_file}")
        print(f"GLiNER training data: {gliner_file}")
        print(f"Statistics: {stats_file}")

def main():
    """Main entry point"""
    try:
        generator = BIQuestionGenerator()
        generator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
