import spacy

nlp = spacy.blank("en")

question = "What is the total discount amount offered during the clearance sale last quarter?"
entities = ["discount amount", "last quarter"]

doc = nlp(question)
results = []

for ent_text in entities:
    # simple lowercase match
    for token_start in range(len(doc)):
        for token_end in range(token_start + 1, len(doc) + 1):
            span = doc[token_start:token_end]
            if span.text.lower() == ent_text.lower():
                results.append([span.start_char, span.end_char, "MEASURE" if ent_text=="discount amount" else "TIMEFRAME"])
                break

print(results)
# Output: [[12, 27, 'MEASURE'], [68, 80, 'TIMEFRAME']]
