import anthropic
import json

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
# 2 - Read your JSONL file (each line should be a valid JSON dict)
with open("part_ae", "r") as f:
    requests = [json.loads(line) for line in f]

# 3 - Process each request sequentially
results = []
for i, req in enumerate(requests, 1):
    # Example request format must include a 'messages' field or 'prompt'
    # Here we assume the same format as OpenAI chat completions
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        messages=req["params"]["messages"],
    )

    results.append({
        "request": req,
        "response": response
    })

    print(f"[{i}/{len(requests)}] Done.")

# 4 - Save results
with open("output/claude_results_pref_v2.jsonl", "a") as f:
    for r in results:
        f.write(json.dumps(r, default=str) + "\n")

print("✅ All requests completed and saved to output/claude_results.jsonl")
