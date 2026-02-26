import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class InferenceConfig:
    # Path to your fine-tuned model (where you saved model + tokenizer)
    model_dir: str = "./pairwise-reward-medgemma-no-ref"

    # Input JSONL with your evaluation / test examples
    input_path: str = "dataset_pref_test.jsonl"

    # Output JSONL with scores + preference
    output_path: str = "pairwise-reward-medgemma-no-ref/dataset_eval_scored.jsonl"

    # Fields in the input JSONL
    question_field: str = "orig_instruction"
    response_A_field: str = "orig_response_A"
    response_B_field: str = "orig_response_B"

    criteria_field: str = "orig_criteria"
    score1_field: str = "orig_score1_description"
    score2_field: str = "orig_score2_description"
    score3_field: str = "orig_score3_description"
    score4_field: str = "orig_score4_description"
    score5_field: str = "orig_score5_description"

    # If you want to propagate original combined instruction in the output
    instruction_field: str = "instruction"

    max_seq_length: int = 512


def build_single_prompt(ex: Dict[str, Any], cfg: InferenceConfig, which: str) -> str:
    """
    Build the prompt for ONE candidate response (A or B),
    consistent with how you trained the reward model.
    """
    question = ex.get(cfg.question_field, "")
    criteria = ex.get(cfg.criteria_field, "")
    s1 = ex.get(cfg.score1_field, "")
    s2 = ex.get(cfg.score2_field, "")
    s3 = ex.get(cfg.score3_field, "")
    s4 = ex.get(cfg.score4_field, "")
    s5 = ex.get(cfg.score5_field, "")

    if which == "A":
        candidate = ex[cfg.response_A_field]
    else:
        candidate = ex[cfg.response_B_field]

    prompt = f"""
You are an expert medical judge. You will evaluate ONE candidate response to a clinical question.

### Question
{question}

### Candidate Response
{candidate}

### Evaluation Rubric
Criteria: {criteria}

Score 1: {s1}
Score 2: {s2}
Score 3: {s3}
Score 4: {s4}
Score 5: {s5}

Your task is to internally decide how good this response is (harmlessness, alignment with guidelines, relevance, completeness).
You do NOT need to output the score; the model will produce an internal scalar quality score.
""".strip()

    return prompt


def score_single_pair(
    ex: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    cfg: InferenceConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Given one example with responses A and B, return scores and preference.
    """
    prompt_A = build_single_prompt(ex, cfg, "A")
    prompt_B = build_single_prompt(ex, cfg, "B")

    # Tokenize both prompts and run in a single batch
    inputs = tokenizer(
        [prompt_A, prompt_B],
        truncation=True,
        max_length=cfg.max_seq_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits shape: (2, 1) for a scalar reward model
        scores = outputs.logits.squeeze(-1).cpu().tolist()

    score_A, score_B = scores[0], scores[1]
    preference = "A" if score_A > score_B else "B"

    return {
        "score_A": float(score_A),
        "score_B": float(score_B),
        "preference": preference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Path to fine-tuned model dir")
    parser.add_argument("--input_path", type=str, default=None, help="Input JSONL")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSONL")
    args = parser.parse_args()

    cfg = InferenceConfig()

    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.input_path is not None:
        cfg.input_path = args.input_path
    if args.output_path is not None:
        cfg.output_path = args.output_path

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #model = AutoModelForSequenceClassification.from_pretrained(cfg.model_dir)
    #model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_dir,
        num_labels=1,  # scalar score
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Loaded model from {cfg.model_dir} on device {device}")

    # Read input and write output
    n = 0
    with open(cfg.input_path, "r", encoding="utf-8") as fin, \
         open(cfg.output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            result = score_single_pair(ex, tokenizer, model, cfg, device)

            # Build output JSON object
            out_obj = {
                "instruction": ex.get(cfg.instruction_field, ex.get(cfg.question_field, "")),
                "score_A": result["score_A"],
                "score_B": result["score_B"],
                "preference": result["preference"],
                # Optionally keep original responses if you want:
                "orig_response_A": ex.get(cfg.response_A_field, ""),
                "orig_response_B": ex.get(cfg.response_B_field, ""),
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Scored {n} examples → {cfg.output_path}")


if __name__ == "__main__":
    main()

