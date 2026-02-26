# check_claude_results.py

import json
from pathlib import Path

# --- 1) Read Claude results ---
results_path = Path("output/claude_results.jsonl")
if not results_path.exists():
    raise FileNotFoundError(f"{results_path} not found. Run the generation script first.")

with results_path.open("r", encoding="utf-8") as f:
    api_response = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(api_response)} Claude responses.")
print(json.dumps(api_response[0], indent=2))  # show first for inspection

# --- 2) Token usage aggregation ---
def calculate_tokens(api_response):
    input_tokens = 0
    output_tokens = 0

    for item in api_response:
        usage = item.get("usage", {})
        if usage:
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0

    return input_tokens, output_tokens

input_tokens, output_tokens = calculate_tokens(api_response)
print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")

# --- 3) Pricing ---
# Claude 3.7 Sonnet (as of Oct 2025)
# Source: Anthropic pricing (USD per million tokens)
price_per_million_input = 3.00
price_per_million_output = 15.00

cost_input = input_tokens / 1e6 * price_per_million_input
cost_output = output_tokens / 1e6 * price_per_million_output
total_cost = cost_input + cost_output
n_cases = len(api_response)

print(f"\nCost breakdown for {n_cases} cases:")
print(f"- Input cost:  ${cost_input:.4f}")
print(f"- Output cost: ${cost_output:.4f}")
print(f"- Total cost:  ${total_cost:.4f}")
print(f"- Cost per case: ${total_cost / n_cases:.4f}")

# --- 4) (Optional) estimate cost for entire dataset ---
# from utils import load_data
# df_cases, df_labels, df_paths = load_data()
# total_dataset_cost = (total_cost / n_cases) * len(df_cases)
# print(f"Estimated total dataset cost for {len(df_cases)} cases: ${total_dataset_cost:.2f}")
