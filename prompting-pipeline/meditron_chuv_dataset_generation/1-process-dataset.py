'''The data of interest is a collection of pairs of
* A case description
* The corresponding label, i.e. the final diagnostic
This script aims to structure these pairs into a set of prompts 
aimed to be sent to gpt4.
'''

# Import libraries
import json
output_file = "requests_v3.jsonl"

with open("evaluation_template.txt") as f:
    template = f.read()


# Process data
with open("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/dataset_ref_based_scores_test_new_v2.jsonl") as f,  open(output_file, "w") as f_out:
    for idx, line in enumerate(f, start=1):
        item = json.loads(line)

        # Replace placeholders automatically
        filled_prompt = template
        for key, value in item.items():
            filled_prompt = filled_prompt.replace(f"{{{{ {key} }}}}", str(value))
        
        # Build JSON line with an ID
        json_line = {
            "custom_id": f"req_{idx}",  # <-- Add unique ID
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "OpenMeditron/meditron-CHUV-dpo-bfloat16",
                "messages": [{"role": "user", "content": filled_prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            }
        }
        f_out.write(json.dumps(json_line) + '\n')

print(f"JSONL file '{output_file}' created successfully")
