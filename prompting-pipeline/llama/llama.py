from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

#you should download the weights and put the path here, or use a Hugging Face Hub model if available (but beware of GPU memory requirements)
model_hf = "llama"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_hf)
tokenizer.padding_side = "left"

# Use EOS as pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load pipeline with DeepSpeed config
generator = pipeline(
    "text-generation",
    model=model_hf,                 # pass model name, not a loaded model
    tokenizer=tokenizer,
    device=0,                       # GPU index (0 = first GPU)
    torch_dtype=torch.float16,
    deepspeed="ds_config.json",     # <-- enable DeepSpeed
    batch_size=1,
    max_new_tokens=200
)

# Load template
# you should modify depending on the type of evaluation you want to do (scores or preference)
with open("evaluation_template.txt") as f:
    template = f.read()

# Load all data
# you should modify depending on the type of evaluation you want to do (scores or preference)
data = []
with open("data_pref.jsonl") as f:
    for line in f:
        item = json.loads(line)

        # Replace placeholders
        filled_prompt = template
        for key, value in item.items():
            filled_prompt = filled_prompt.replace(f"{{{{ {key} }}}}", value)

        data.append({
            "instruction": item.get("QUESTION"),
            "response": item.get("RESPONSE"),
            "reference_answer": item.get("GOLD_RESPONSE"),
            "prompt": filled_prompt
        })

# Run inference in batches
prompts = [d["prompt"] for d in data]
outputs = generator(
    prompts,
    max_new_tokens=500,
    temperature=0.1,
    return_full_text=False
)

# Write results
with open("outputs_llama.jsonl", "a") as f_out:
    for d, out in zip(data, outputs):
        d["generated_output"] = out[0]["generated_text"]
        f_out.write(json.dumps(d) + "\n")
