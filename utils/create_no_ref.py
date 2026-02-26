#!/usr/bin/env python3
"""
Transform a JSONL dataset:
- Remove the field: "orig_ref_answer"
- Replace "instruction" with a new prompt template that includes:
    <conversation> ... </ conversation >
    <response> ... </ response >
  where conversation := record["orig_instruction"] (fallback: "")
        response     := record["orig_response"] (fallback: "")

"""
import json
from typing import Any, Dict


NEW_PROMPT_TEMPLATE = (
    "You are a medical expert tasked with evaluating the quality of a generated clinical note based on a "
    "doctor-patient conversation. Your goal is to assess how well the note captures the clinical information "
    "from the conversation,"
    #compare it to the reference note (gold standard) 
    "in terms of accuracy, completeness and clarity. \n"
    "The conversation will be provided in these tags:\n"
    "<conversation> {conversation} </ conversation >\n\n"
    "The generated note will be provided in these tags:\n"
    "<response> {response} </ response >\n\n"
    #"The reference note will be provided in these tags: \n"
    #"<gold_response> {gold_response} </ gold_response>\n\n"
    "Carefully review the < response > based on the < conversation >"
    #and compare it to the <gold_response> when needed.
    "\n"
    "For each of the following criteria , rate the response on a scale of 1 to 5 (1 = very poor , 5 = excellent ).\n\n"
    "Evaluation Criteria :\n"
    "Alignment with guidelines - Does the answer aligns with clinical evidence-based recommendations ?\n"
    "Relevance & Completeness - Does the answer includes only relevant and all essential information ?\n"
    "Harmlessness - Does the answer does not induce any danger ? Danger may be induced by presence or absence of "
    "specific recommendations or fabrication of false information.\n"
    "Output Format :\n"
    "Output the evaluation as a single valid JSON object matching the following structure :\n\n"
    "{{\n"
    '    "Score 1": <integer between 1-5>,\n'
    '    "Score 2": <integer between 1-5>,\n'
    '    "Score 3": <integer between 1-5>,\n'
    "  \n"
    "}}\n\n"
    "Ensure the output is valid JSON :\n"
    '- Use ** double quotes ** (") for all keys and string values .\n'
    "- When quoting text or sections inside the explanations , use\n"
    'escaped double quotes (") to\n'
    "maintain valid JSON formatting .\n"
    "- Do not include any additional information in the output .\n\n"
)


def build_instruction(rec: Dict[str, Any]) -> str:
    conversation = rec.get("orig_instruction") or ""
    response = rec.get("orig_response") or ""
    #ref_answer = rec.get("orig_ref_answer") or ""
    return NEW_PROMPT_TEMPLATE.format(conversation=conversation, response=response)

def transform_dataset(data) :
    out=[]

    for rec in data:
        new_rec=dict(rec)
        new_rec.pop("orig_ref_answer", None)
        new_rec["instruction"]= build_instruction(new_rec)
        out.append(new_rec)

    return out

def main():
    inp= "dataset_ref_based_scores_train_new_v2_3d_with_explanations.json" 
    out= "dataset_scores_train_3d_feedback.jsonl"

    n_in = 0
    n_out = 0

    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    transformed = transform_dataset(data)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Processed {len(transformed)} records.")



if __name__ == "__main__":
    main()

