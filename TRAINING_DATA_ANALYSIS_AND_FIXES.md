# Training Data Analysis and Fixes

## Executive Summary

After manually reviewing both training datasets (`more_training_data.json` and `bi_training_data.json`), I've identified several critical issues that need to be addressed before training your SLM. The training data has inconsistencies with the expected output format specified in the prompt.txt file.

## Issues Found and Fixed

### ✅ **FIXED ISSUES in more_training_data.json**

1. **Missing Timeframe in Q3 2024 Example**
   - **Input**: "Show me annual recurring revenue for Q3 2024"
   - **Problem**: Missing timeframe with "Q3 2024" phrase
   - **Fix**: Added `"timeframe": {"phrase": "Q3 2024"}`

2. **Redundant Dimensions and Breakdowns**
   - **Input**: "Show me the breakdown of sales by product category and region"
   - **Problem**: Both dimensions and breakdowns had the same items
   - **Fix**: Removed redundant dimensions, kept only breakdowns

3. **Missing Unmatched Intents for Business Terms**
   - **Input**: "What is the average for premium customers?"
   - **Problem**: "premium customers" should be in unmatched_intents
   - **Fix**: Moved "premium customers" to unmatched_intents as business_entity

### ❌ **CRITICAL ISSUES in bi_training_data.json**

1. **Incorrect Sub-Question Transformations**
   - **Input**: "How many heads of the publishers are older than 56 ?"
   - **Output sub_question**: "How many heads of the departments are older than 56 ?"
   - **Problem**: "publishers" was incorrectly changed to "departments"
   - **Impact**: This will train the model to incorrectly transform user input

2. **Incorrect Measure Names for Count Questions**
   - **Input**: "How many heads of the publishers are older than 56 ?"
   - **Current**: `"name": "Total", "calculation": "Count"`
   - **Problem**: Should have `"name": "Count"` for count operations
   - **Impact**: Inconsistent measure naming

3. **Poor Unmatched Intents Handling**
   - Many business terms marked as "unknown_term" when they should be "business_entity"
   - Individual words split instead of keeping meaningful phrases together
   - **Impact**: Will not properly capture business concepts for knowledge base mapping

4. **Inconsistent Entity Mapping**
   - "revenue" mapped to "budget" in some examples
   - "publishers" mapped to "departments" 
   - **Impact**: Confusing the model about entity relationships

## Additional Issues Found

### **Pattern Recognition Problems**

1. **Missing Complex Question Detection**
   - Some questions that should be marked as complex are treated as simple
   - Missing proper step breakdown for multi-step questions

2. **Inconsistent Calculation Types**
   - Some measures use "Total" when they should use "Average" or "Count"
   - Missing proper calculation type mapping

3. **Breakdown vs Dimension Confusion**
   - Some examples incorrectly use dimensions when they should use breakdowns
   - Inconsistent handling of "by" phrases

### **Data Quality Issues**

1. **Duplicate Examples**
   - Found identical examples with different outputs
   - Inconsistent handling of similar questions

2. **Missing Edge Cases**
   - No examples of ambiguous terms requiring human input
   - Missing examples of request_human_input intent

## Recommendations Before Training

### **1. Fix bi_training_data.json (CRITICAL)**

The bi_training_data.json file has fundamental issues that will severely impact training:

- **Fix all sub_question transformations** to match the original input
- **Correct measure names** for count operations
- **Improve unmatched_intents handling** to capture meaningful business phrases
- **Standardize entity mapping** to be consistent

### **2. Add Missing Examples**

Based on the prompt.txt requirements, you need examples for:

- **request_human_input intent** for ambiguous terms
- **Complex questions with proper step breakdown**
- **Edge cases** like multiple filter values, time-based dependencies
- **Comparative analysis** questions

### **3. Validate Against Prompt Requirements**

Ensure all examples follow the exact format specified in prompt.txt:

- **Complexity detection** working correctly
- **Proper step breakdown** for multi-step questions
- **Correct unmatched_intents** for business terms
- **Consistent timeframe and timegrain** handling

### **4. Data Consistency**

- **Remove duplicates** and ensure consistent handling
- **Standardize naming conventions** for measures and dimensions
- **Validate JSON structure** for all examples

## Immediate Action Required

1. **DO NOT TRAIN** with the current bi_training_data.json - it will create a broken model
2. **Fix all sub_question transformations** in bi_training_data.json
3. **Add missing examples** for request_human_input intent
4. **Validate all examples** against the prompt.txt requirements
5. **Test with a small subset** before full training

## Quality Checklist

Before training, ensure every example:

- [ ] Sub_question matches the input (no incorrect transformations)
- [ ] Proper measure names and calculations
- [ ] Correct unmatched_intents for business terms
- [ ] Follows the exact JSON structure from prompt.txt
- [ ] Handles complexity detection correctly
- [ ] Includes examples of all required intents

## Conclusion

The more_training_data.json file is in better shape after the fixes, but bi_training_data.json requires significant work. The current state will not produce a reliable SLM. Focus on fixing the fundamental issues in bi_training_data.json before proceeding with training.
