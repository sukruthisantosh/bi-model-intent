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
prompt_path = os.path.join(os.path.dirname(__file__), "../resources/prompts/prompt.txt")
model_dir = os.path.join(os.path.dirname(__file__), "../model")

questions = [
    "What are the ad impressions rendered and shown for last one month for all publishers for India?",
    "For the perf data, i need the sum of valid ad requests, total ad requests, CAS FORWARDS, CAS fills, CAS wins and total burn for the last QTD for the PMP revenue channel.",
    "Which are the top 5 publishers in India with the highest number of ad impressions rendered in the last month?",
    "What is the percentage contribution of total ad requests for Zynga and Easybrain when limit ad tracking is true over the last 30 days?",
    "Provide a day-level trend of total burn and clicks for the bidder \"FeeltapmediaRTBD\" over the last 15 days."
]

# Load prompt and context
prompt_template = load_prompt_template(prompt_path)
context = load_context_files(model_dir)

# Loop over questions
for i, question in enumerate(questions, 1):
    full_prompt = build_prompt(prompt_template, question, context)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for BI intent discovery."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    print(f"\n=== Question {i} ===")
    print(f"Q: {question}")
    print(f"A: {response.choices[0].message.content}")
