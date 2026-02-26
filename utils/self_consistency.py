###
# This script reads multiple JSONL files containing model outputs, extracts the scores from the raw generations, computes the mean score for each prompt, and writes the merged results to a new JSONL file.
import json
from pathlib import Path
from statistics import mean, median, median_high
import re
# Paths to the 5 input files
input_files = [
    "output_with_generated-v5.11.jsonl",
    "output_with_generated-v5.12.jsonl",
    "output_with_generated-v5.13.jsonl",
    "output_with_generated-v5.14.jsonl",
    "output_with_generated-v5.15.jsonl",
    "output_with_generated-v5.16.jsonl",
    "output_with_generated-v5.17.jsonl",
    "output_with_generated-v5.18.jsonl",
    "output_with_generated-v5.19.jsonl",
    "output_with_generated-v5.20.jsonl",
]

output_file = "merged_mean_outputs_feedback.jsonl"

# Open all files
file_handles = [open(f, "r") for f in input_files]

with open(output_file, "w") as out_f:
    for lines in zip(*file_handles):
        records = [json.loads(line) for line in lines]

        prompt = records[0]["instruction"]
        #outputs = [float(r["output"]) for r in records]
        #outputs = [float(re.findall(r'"Score"\s*:\s*([-+]?\d*\.\d+|\d+)', r["raw_generation"])[0]) for r in records]
        outputs = [
            float(match[0]) if (match := re.findall(r'"Score"\s*:\s*([-+]?\d*\.\d+|\d+)', r["raw_generation"])) 
            else 3
            for r in records
        
            ]
        valid_outputs = [o for o in outputs if o is not None]
        if valid_outputs:
            merged_record = {
                "instruction": prompt,
                "orig_response": records[0]["orig_response"],
                "mean_generated_output": mean(valid_outputs),
                "all_outputs": outputs
            }

        out_f.write(json.dumps(merged_record) + "\n")

# Close files
for f in file_handles:
    f.close()

