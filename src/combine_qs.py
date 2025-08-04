import os

# Paths
existing_path = os.path.join(os.path.dirname(__file__), "../resources/existing_questions.txt")
generated_path = os.path.join(os.path.dirname(__file__), "../resources/generated_questions.txt")
combined_path = os.path.join(os.path.dirname(__file__), "../resources/all_questions.txt")

# Load existing questions
with open(existing_path, "r") as f:
    existing_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("//")]

# Load generated questions
with open(generated_path, "r") as f:
    generated_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("//")]

# Remove numbering from generated questions if present
import re
def strip_number(line):
    return re.sub(r"^\d+\.\s*", "", line)

generated_clean = [strip_number(line) for line in generated_lines]

# Renumber generated questions starting from 61
generated_numbered = [f"{i+61}. {q}" for i, q in enumerate(generated_clean)]

# Combine and write to new file
with open(combined_path, "w") as f:
    for line in existing_lines:
        f.write(line + "\n")
    for line in generated_numbered:
        f.write(line + "\n")

print(f"Combined questions written to {combined_path}")