import os
import openai

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_file(path):
    with open(path, "r") as f:
        return f.read()

def load_context_files(model_dir):
    context = ""
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".json"):
            with open(fpath, "r") as f:
                context += f"\n\n[{fname}]\n{f.read()}"
    return context

def build_generation_prompt(prompt_template, model_context):
    return prompt_template.replace("{{ model_context }}", model_context)

# --- Script usage below ---

# Paths
prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompts/question_generation_prompt.txt")
model_dir = os.path.join(os.path.dirname(__file__), "../model")

# Load prompt and model context
prompt_template = load_file(prompt_path)
model_context = load_context_files(model_dir)
full_prompt = build_generation_prompt(prompt_template, model_context)

# Call OpenAI API (GPT-4o, change model as needed)
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for BI question generation."},
        {"role": "user", "content": full_prompt}
    ],
    temperature=0.3,
    max_tokens=2048
)

output = response.choices[0].message.content
print(output)

# Save output to a new file under resources
output_path = os.path.join(os.path.dirname(__file__), "../resources/generated_questions.txt")
with open(output_path, "w") as f:
    f.write(output)
