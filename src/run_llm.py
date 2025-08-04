import os
import openai

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load prompt template
def load_prompt_template(prompt_path):
    with open(prompt_path, "r") as f:
        return f.read()

# Load context files (e.g., schema, model info)
def load_context_files(model_dir):
    context = ""
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".json"):
            with open(fpath, "r") as f:
                context += f"\n\n[{fname}]\n{f.read()}"
    return context

# Compose the full prompt
def build_prompt(prompt_template, question, context):
    return prompt_template.replace("{{ context.current_question }}", question).replace("{{ context.model_files }}", context)

# --- Script usage below ---

# Paths
prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompt.txt")
model_dir = os.path.join(os.path.dirname(__file__), "../model")
question = "What are the ad impressions rendered and shown for last one month for all publishers for India?"

# Load prompt and context
prompt_template = load_prompt_template(prompt_path)
context = load_context_files(model_dir)
full_prompt = build_prompt(prompt_template, question, context)

# Call OpenAI API (GPT-4o, change model as needed)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for BI intent discovery."},
        {"role": "user", "content": full_prompt}
    ],
    temperature=0.2,
    max_tokens=1024
)

print(response.choices[0].message.content)