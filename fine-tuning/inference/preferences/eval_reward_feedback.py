#!/usr/bin/env python3
"""
Inference for a "reward model with feedback" trained via SFT to generate preference + feedback.

Input: JSONL with same shape as training set (at least):
  - instruction
  - orig_response_A
  - orig_response_B
(Other fields are preserved.)

Output: JSONL with added fields:
  - pred_winner: "A" or "B" (or None if parse failed)
  - pred_score: float in [0,1] where 1 => A preferred, 0 => B preferred (or None)
  - pred_strength: int in [1,5] (or None)
  - pred_feedback: str (or None)
  - pred_raw: raw generated text

Usage:
  python infer_reward_feedback.py \
    --model_dir /path/to/your_finetuned_checkpoint \
    --input_jsonl /path/to/input.jsonl \
    --out_jsonl /path/to/output_scored.jsonl \
    --max_new_tokens 256 \
    --batch_size 1
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=False, default="pref-reward-feedback-v2" ,help="Path or HF id of the fine-tuned model checkpoint.")
    p.add_argument("--input_jsonl", type=str, required=False, default="dataset_ref_based_pref_test.jsonl")
    p.add_argument("--out_jsonl", type=str, required=False, default="pref-reward-feedback-v2/inference.jsonl")

    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--response_a_key", type=str, default="orig_response_A")
    p.add_argument("--response_b_key", type=str, default="orig_response_B")

    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument("--temperature", type=float, default=0.0) 
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--device_map", type=str, default="auto") 
    return p.parse_args()


def build_prompt(instruction: str, resp_a: str, resp_b: str) -> str:
    """
    Prompt the model to behave like a preference judge and output winner/score/feedback.
    We keep the schema explicit to make parsing reliable.
    """
    instruction = (instruction or "").strip()
    resp_a = (resp_a or "").strip()
    resp_b = (resp_b or "").strip()

    return (
        "You are an expert judge. Compare Response A and Response B for the given instruction.\n"
        "Choose the better response and provide a score and feedback.\n\n"
        "Output format (exactly):\n"
        "Winner: A or B\n"
        "Score: a number in [0,1] where 1 means A is better and 0 means B is better\n"
        "Strength: an integer 1-5 (1=weak preference, 5=strong preference)\n"
        "Feedback: a short explanation comparing A vs B\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        "Winner:"
    )


_WIN_RE = re.compile(r"(?im)^\s*winner\s*:\s*([ab])\s*$")
_SCORE_RE = re.compile(r"(?im)^\s*score\s*:\s*([-+]?\d+(\.\d+)?)\s*$")
_STRENGTH_RE = re.compile(r"(?im)^\s*strength\s*:\s*(\d+)\s*$")
_FEEDBACK_RE = re.compile(r"(?is)^\s*feedback\s*:\s*(.*)\s*$")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_generation(text: str) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[str]]:
    """
    Parse Winner / Score / Strength / Feedback from model output.
    Returns (winner, score, strength, feedback).
    """
    winner = None
    score = None
    strength = None
    feedback = None

    m = _WIN_RE.search(text)
    if m:
        winner = m.group(1).upper()

    m = _SCORE_RE.search(text)
    if m:
        try:
            score = float(m.group(1))
            score = clamp(score, 0.0, 1.0)
        except Exception:
            score = None

    m = _STRENGTH_RE.search(text)
    if m:
        try:
            strength = int(m.group(1))
            strength = max(1, min(5, strength))
        except Exception:
            strength = None

    m = _FEEDBACK_RE.search(text)
    if m:
        feedback = m.group(1).strip()

    # If winner exists but score missing, derive a default.
    if winner in ("A", "B") and score is None:
        score = 1.0 if winner == "A" else 0.0

    # If score exists but winner missing, infer winner.
    if score is not None and winner is None:
        winner = "A" if score >= 0.5 else "B"

    return winner, score, strength, feedback


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    # Place inputs on the first device used by the model
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    do_sample = temperature is not None and temperature > 0.0

    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated portion for cleaner parsing:
    # We slice off the prompt length per row.
    gen_texts = []
    for i in range(out_ids.size(0)):
        prompt_len = enc["input_ids"][i].shape[0]
        gen_part = out_ids[i][prompt_len:]
        gen_texts.append(tokenizer.decode(gen_part, skip_special_tokens=True).strip())

    return gen_texts


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
    )
    model.eval()

    # Clear output if exists
    if out_path.exists():
        out_path.unlink()

    batch_rows: List[Dict[str, Any]] = []
    batch_prompts: List[str] = []

    def flush():
        nonlocal batch_rows, batch_prompts
        if not batch_rows:
            return

        gens = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        with out_path.open("a", encoding="utf-8") as fout:
            for row, gen in zip(batch_rows, gens):
                winner, score, strength, feedback = parse_generation(gen)

                row_out = dict(row)
                row_out["pred_winner"] = winner
                row_out["pred_score"] = score
                row_out["pred_strength"] = strength
                row_out["pred_feedback"] = feedback
                row_out["pred_raw"] = gen

                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")

        batch_rows, batch_prompts = [], []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            instr = row.get(args.instruction_key, "")
            ra = row.get(args.response_a_key, "")
            rb = row.get(args.response_b_key, "")

            prompt = build_prompt(instr, ra, rb)

            batch_rows.append(row)
            batch_prompts.append(prompt)

            if len(batch_rows) >= args.batch_size:
                flush()

    flush()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

