#!/usr/bin/env python3
"""
Analyze BI Training Data Quality
================================

This script analyzes the converted BI training data to assess quality,
coverage, and readiness for training.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Any

def analyze_bi_training_data():
    """Analyze the converted BI training data"""
    
    print("Loading BI training data...")
    with open("bi_training_data.json", "r") as f:
        bi_data = json.load(f)
    
    print(f"Analyzing {len(bi_data)} BI training examples...")
    
    # Analysis results
    analysis = {
        "total_examples": len(bi_data),
        "complexity_distribution": {},
        "bi_terms_coverage": {},
        "question_patterns": {},
        "output_quality": {},
        "issues": []
    }
    
    # 1. Complexity Distribution Analysis
    print("\n1. Analyzing complexity distribution...")
    step_counts = {"step_1": 0, "step_2": 0, "step_3": 0}
    
    for example in bi_data:
        output = example["output"]
        if "discovery_results" in output:
            for result in output["discovery_results"]:
                step_id = result.get("step_id", "unknown")
                if step_id in step_counts:
                    step_counts[step_id] += 1
    
    analysis["complexity_distribution"] = {
        "simple_questions": step_counts["step_1"],
        "complex_questions": step_counts["step_2"] + step_counts["step_3"],
        "multi_step_questions": step_counts["step_2"],
        "three_step_questions": step_counts["step_3"],
        "complexity_ratio": (step_counts["step_2"] + step_counts["step_3"]) / len(bi_data)
    }
    
    # 2. BI Terms Coverage Analysis
    print("2. Analyzing BI terms coverage...")
    bi_terms = [
        "publishers", "advertisers", "bidders", "campaigns", "users", "customers", 
        "viewers", "clicks", "revenue", "spend", "cost", "budget", "websites", 
        "apps", "channels", "sites", "visitors", "audience", "ads", "creatives", 
        "placements", "impressions", "conversions", "requests", "bids", "locations", 
        "regions", "markets", "territories", "videos", "content", "activities", 
        "events", "managers", "leaders", "operators", "controllers"
    ]
    
    term_counts = Counter()
    for example in bi_data:
        question = example["input"].lower()
        for term in bi_terms:
            if term in question:
                term_counts[term] += 1
    
    analysis["bi_terms_coverage"] = {
        "total_bi_terms_used": len([term for term, count in term_counts.items() if count > 0]),
        "most_common_terms": term_counts.most_common(10),
        "terms_with_zero_usage": [term for term in bi_terms if term_counts[term] == 0]
    }
    
    # 3. Question Patterns Analysis
    print("3. Analyzing question patterns...")
    patterns = {
        "how_many": 0, "list": 0, "what_are": 0, "compare": 0, "top": 0, 
        "bottom": 0, "average": 0, "maximum": 0, "minimum": 0, "trend": 0,
        "percentage": 0, "which": 0, "show": 0, "find": 0, "return": 0
    }
    
    for example in bi_data:
        question = example["input"].lower()
        if "how many" in question:
            patterns["how_many"] += 1
        if "list" in question:
            patterns["list"] += 1
        if "what are" in question:
            patterns["what_are"] += 1
        if "compare" in question or "versus" in question or "vs" in question:
            patterns["compare"] += 1
        if "top" in question:
            patterns["top"] += 1
        if "bottom" in question or "lowest" in question:
            patterns["bottom"] += 1
        if "average" in question:
            patterns["average"] += 1
        if "maximum" in question or "highest" in question:
            patterns["maximum"] += 1
        if "minimum" in question or "lowest" in question:
            patterns["minimum"] += 1
        if "trend" in question or "change" in question:
            patterns["trend"] += 1
        if "percentage" in question or "%" in question:
            patterns["percentage"] += 1
        if "which" in question:
            patterns["which"] += 1
        if "show" in question:
            patterns["show"] += 1
        if "find" in question:
            patterns["find"] += 1
        if "return" in question:
            patterns["return"] += 1
    
    analysis["question_patterns"] = patterns
    
    # 4. Output Quality Analysis
    print("4. Analyzing output quality...")
    output_issues = []
    valid_outputs = 0
    
    for i, example in enumerate(bi_data):
        output = example["output"]
        
        # Check basic structure
        if "intent" not in output:
            output_issues.append(f"Example {i}: Missing intent field")
            continue
        
        if "discovery_results" not in output:
            output_issues.append(f"Example {i}: Missing discovery_results")
            continue
        
        # Check discovery results structure
        for j, result in enumerate(output["discovery_results"]):
            required_fields = ["step_id", "sub_question", "measures", "dimensions", 
                             "timegrain", "timeframe", "pattern", "segments", 
                             "breakdowns", "unmatched_intents"]
            
            for field in required_fields:
                if field not in result:
                    output_issues.append(f"Example {i}, Result {j}: Missing {field}")
        
        valid_outputs += 1
    
    analysis["output_quality"] = {
        "valid_outputs": valid_outputs,
        "output_issues": output_issues,
        "validity_rate": valid_outputs / len(bi_data)
    }
    
    # 5. Identify Issues
    print("5. Identifying issues...")
    issues = []
    
    # Check for remaining non-BI terms
    non_bi_terms = ["departments", "employees", "students", "schools", "companies", 
                   "customers", "films", "movies", "books", "sports", "games"]
    
    for example in bi_data:
        question = example["input"].lower()
        for term in non_bi_terms:
            if term in question:
                issues.append(f"Non-BI term found: '{term}' in question")
                break
    
    # Check for inconsistent terminology
    if "departments" in str(bi_data).lower():
        issues.append("Inconsistent terminology: 'departments' still present in outputs")
    
    analysis["issues"] = issues
    
    return analysis

def print_analysis_report(analysis):
    """Print comprehensive analysis report"""
    
    print("\n" + "="*80)
    print("BI TRAINING DATA ANALYSIS REPORT")
    print("="*80)
    
    # 1. Overall Statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total examples: {analysis['total_examples']:,}")
    print(f"   Valid outputs: {analysis['output_quality']['valid_outputs']:,}")
    print(f"   Validity rate: {analysis['output_quality']['validity_rate']:.1%}")
    
    # 2. Complexity Distribution
    print(f"\n🎯 COMPLEXITY DISTRIBUTION:")
    comp = analysis['complexity_distribution']
    print(f"   Simple questions: {comp['simple_questions']:,} ({comp['simple_questions']/analysis['total_examples']:.1%})")
    print(f"   Complex questions: {comp['complex_questions']:,} ({comp['complex_questions']/analysis['total_examples']:.1%})")
    print(f"   Multi-step questions: {comp['multi_step_questions']:,}")
    print(f"   Three-step questions: {comp['three_step_questions']:,}")
    print(f"   Complexity ratio: {comp['complexity_ratio']:.1%}")
    
    # 3. BI Terms Coverage
    print(f"\n🔍 BI TERMS COVERAGE:")
    terms = analysis['bi_terms_coverage']
    print(f"   BI terms used: {terms['total_bi_terms_used']}")
    print(f"   Most common terms:")
    for term, count in terms['most_common_terms'][:5]:
        print(f"     {term}: {count:,} times")
    
    if terms['terms_with_zero_usage']:
        print(f"   Terms with zero usage: {len(terms['terms_with_zero_usage'])}")
    
    # 4. Question Patterns
    print(f"\n📝 QUESTION PATTERNS:")
    patterns = analysis['question_patterns']
    total = analysis['total_examples']
    
    pattern_percentages = []
    for pattern, count in patterns.items():
        if count > 0:
            percentage = count / total
            pattern_percentages.append((pattern, count, percentage))
    
    pattern_percentages.sort(key=lambda x: x[1], reverse=True)
    
    for pattern, count, percentage in pattern_percentages[:10]:
        print(f"   {pattern}: {count:,} ({percentage:.1%})")
    
    # 5. Output Quality
    print(f"\n✅ OUTPUT QUALITY:")
    quality = analysis['output_quality']
    print(f"   Valid outputs: {quality['valid_outputs']:,}")
    print(f"   Validity rate: {quality['validity_rate']:.1%}")
    
    if quality['output_issues']:
        print(f"   Issues found: {len(quality['output_issues'])}")
        for issue in quality['output_issues'][:5]:
            print(f"     - {issue}")
        if len(quality['output_issues']) > 5:
            print(f"     ... and {len(quality['output_issues']) - 5} more")
    
    # 6. Issues
    print(f"\n⚠️  ISSUES IDENTIFIED:")
    if analysis['issues']:
        unique_issues = list(set(analysis['issues']))
        for issue in unique_issues[:10]:
            print(f"   - {issue}")
        if len(unique_issues) > 10:
            print(f"   ... and {len(unique_issues) - 10} more")
    else:
        print("   No major issues found!")
    
    # 7. Quality Score
    print(f"\n🎯 QUALITY SCORE:")
    
    # Calculate quality score
    score = 0
    max_score = 100
    
    # Volume (20 points)
    if analysis['total_examples'] >= 7000:
        score += 20
    elif analysis['total_examples'] >= 5000:
        score += 15
    elif analysis['total_examples'] >= 3000:
        score += 10
    
    # Complexity distribution (20 points)
    comp_ratio = analysis['complexity_distribution']['complexity_ratio']
    if 0.3 <= comp_ratio <= 0.7:
        score += 20
    elif 0.2 <= comp_ratio <= 0.8:
        score += 15
    else:
        score += 10
    
    # BI terms coverage (20 points)
    bi_terms_used = analysis['bi_terms_coverage']['total_bi_terms_used']
    if bi_terms_used >= 20:
        score += 20
    elif bi_terms_used >= 15:
        score += 15
    elif bi_terms_used >= 10:
        score += 10
    
    # Output validity (20 points)
    validity_rate = analysis['output_quality']['validity_rate']
    if validity_rate >= 0.95:
        score += 20
    elif validity_rate >= 0.90:
        score += 15
    elif validity_rate >= 0.80:
        score += 10
    
    # Issues (20 points)
    issue_count = len(analysis['issues'])
    if issue_count == 0:
        score += 20
    elif issue_count <= 10:
        score += 15
    elif issue_count <= 50:
        score += 10
    else:
        score += 5
    
    print(f"   Overall Quality Score: {score}/{max_score} ({score/max_score:.1%})")
    
    if score >= 80:
        print("   🟢 EXCELLENT - Ready for training!")
    elif score >= 60:
        print("   🟡 GOOD - Minor improvements needed")
    else:
        print("   🔴 NEEDS WORK - Significant improvements required")

def main():
    """Run the analysis"""
    
    print("BI Training Data Quality Analysis")
    print("="*50)
    
    # Run analysis
    analysis = analyze_bi_training_data()
    
    # Print report
    print_analysis_report(analysis)
    
    # Save analysis
    with open("bi_training_data_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: bi_training_data_analysis.json")

if __name__ == "__main__":
    main()
