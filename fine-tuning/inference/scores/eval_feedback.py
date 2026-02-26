#!/usr/bin/env python3
"""
Inference script to generate feedback and scores using a fine-tuned LLM. Works with both 3 scores and single score formats.
"""
import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for generating feedback and scores using a fine-tuned LLM.")
    parser.add_argument("--model_path", type=str, default="judge_json_model-no-ref-3d", help="Path to the fine-tuned model.")
    parser.add_argument("--input_jsonl", type=str, default="dataset_scores_test_3d.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, default="judge_json_model-no-ref-3d/output_with_generated.jsonl", help="Path to the output JSONL file.")

    return parser.parse_args()


DEVICE = "cuda"          
MAX_NEW_TOKENS = 150
DTYPE = torch.bfloat16
# =========================


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from text."""
    text = text.strip()

    # Fast path
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Non-greedy first
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Greedy fallback
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def clamp_int_score(x: Any) -> Optional[int]:
    try:
        v = int(x)
        return max(1, min(5, v))
    except Exception:
        return None


def build_prompt(instruction: str) -> str:
    """
    Prompt ends with a hard boundary marker to reduce prompt-echo at inference.
    """
    instr = (instruction or "").strip()

    # If your `instruction` already contains the whole template + response + ref + rubric,
    # this just adds a consistent "now output JSON" instruction and a marker.
    prompt = (
        instr
        + "\n\n"
          "Return JSON only. No markdown. No extra text.\n"
          #"Schema: {\"feedback\":\"...\",\"scores\":{\"Score 1\":<1-5>}, \"Score 2\":<1-5>, \"Score 3\":<1-5>}\n"
          "Schema: {\"feedback\":\"...\",\"scores\":{\"Score\":<1-5>}}\n"
	  "### JSON:\n"
    )
    return prompt


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len= inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eos_ids.append(106)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=1,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    	num_return_sequences=1,
        return_dict_in_generate=True,
    output_scores=True,
    )

    seq = out.sequences[0]
    gen_ids= seq[prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    if device.type == "cpu":
        model.to(device)

    total = 0
    parse_fail = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            instruction = ex.get("instruction", "")

            prompt = build_prompt(instruction)
            gen_text = generate_one(model, tokenizer, prompt)
            parsed = extract_first_json(gen_text)

            gen_feedback = None
            gen_score = None
            parse_ok = False
          

            if isinstance(parsed, dict):
                gen_feedback = parsed.get("feedback")

                scores = parsed.get("scores")
                if isinstance(scores, dict):
                    gen_score = clamp_int_score(scores.get("Score"))

                # fallback if Score is at top-level
                if gen_score is None and "Score" in parsed:
                    gen_score = clamp_int_score(parsed.get("Score"))

                parse_ok = gen_feedback is not None and gen_score is not None

            if not parse_ok:
                parse_fail += 1

            out_ex = {
                "instruction": instruction,
                "generated_feedback": gen_feedback,
                "generated_score": gen_score,
                "parse_ok": parse_ok,
                "raw_generation": gen_text,
            }

            # Keep original fields if present
            for k in [
                "orig_instruction",
                "orig_response",
                "orig_ref_answer",
                "output",
                "orig_score",
                "orig_feedback",
            ]:
                if k in ex:
                    out_ex[k] = ex[k]

            fout.write(json.dumps(out_ex, ensure_ascii=False) + "\n")
            total += 1

    print(
        f"Done. {total} examples written.\n"
        f"Parse failures: {parse_fail} ({(parse_fail / max(total,1)) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()

