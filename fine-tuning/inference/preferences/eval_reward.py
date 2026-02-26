#!/usr/bin/env python3
"""
Reward model inference for MedGemma 27B (SeqCls num_labels=1) + LoRA adapters.

Examples
--------
1) Score a single response:
  python rm_infer.py \
    --base_model google/medgemma-27b-text-it \
    --adapter_dir ./rm-medgemma27b \
    --instruction "..." \
    --response "..."

2) Compare A vs B:
  python rm_infer.py \
    --base_model google/medgemma-27b-text-it \
    --adapter_dir ./rm-medgemma27b \
    --instruction "..." \
    --response_a "..." \
    --response_b "..."

3) Batch JSONL (expects keys: instruction, response_a, response_b):
  python rm_infer.py \
    --base_model google/medgemma-27b-text-it \
    --adapter_dir ./rm-medgemma27b \
    --input_jsonl data.jsonl \
    --out_jsonl scored.jsonl \
    --instruction_key instruction \
    --response_a_key orig_response_A \
    --response_b_key orig_response_B
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="pref-reward-model-v11", help="HF base model id.")
    #p.add_argument("--adapter_dir", type=str, required=True, help="Directory with LoRA adapters + tokenizer files.")
    p.add_argument("--max_length", type=int, default=2048)

    # Single / compare mode
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--response", type=str, default=None)
    p.add_argument("--response_a", type=str, default=None)
    p.add_argument("--response_b", type=str, default=None)

    # Batch mode
    p.add_argument("--input_jsonl", type=str, default="dataset_ref_based_pref_test.jsonl")
    p.add_argument("--out_jsonl", type=str, default="pref-reward-model-v11/inference.jsonl")
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--response_a_key", type=str, default="orig_response_A")
    p.add_argument("--response_b_key", type=str, default="orig_response_B")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def format_prompt(instruction: str) -> str:
    instruction = (instruction or "").strip()
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def load_model_and_tokenizer(base_model: str):
    # QLoRA-style load (4-bit). If you trained differently, set load_in_4bit=False.
    #bnb_config = BitsAndBytesConfig(
     #   load_in_4bit=True,
      #  bnb_4bit_quant_type="nf4",
       # bnb_4bit_compute_dtype=torch.bfloat16,
        #bnb_4bit_use_double_quant=True,
    #)

    # tokenizer: prefer adapter_dir (it may contain pad/eos config)
    tok = AutoTokenizer.from_pretrained(base_model,use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        #quantization_config=bnb_config,
        device_map="auto",
    )
    #model = PeftModel.from_pretrained(base, adapter_dir)
    base.eval()
    return base, tok


@torch.no_grad()
def score_texts(model, tokenizer, prompts: List[str], responses: List[str], max_length: int) -> torch.Tensor:
    """
    Returns a tensor of shape (N,) with scalar rewards.
    We score: [prompt + response]
    """
    assert len(prompts) == len(responses)
    texts = [(p + (r or "").strip()) for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    # Move to model device(s). device_map="auto" => enc tensors must be on first device.
    # The safe approach is to move to model.device if single-device;
    # for multi-device, transformers handles internally, but inputs should be on the first device.
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    out = model(**enc)
    # logits shape: (N, 1)
    rewards = out.logits.squeeze(-1).float().cpu()
    return rewards


def compare(model, tokenizer, instruction: str, a: str, b: str, max_length: int):
    prompt = format_prompt(instruction)
    rewards = score_texts(model, tokenizer, [prompt, prompt], [a, b], max_length)
    ra, rb = float(rewards[0]), float(rewards[1])
    winner = "A" if ra >= rb else "B"
    return ra, rb, winner


def run_single(args, model, tokenizer):
    if args.response is not None:
        prompt = format_prompt(args.instruction or "")
        r = score_texts(model, tokenizer, [prompt], [args.response], args.max_length)
        print(json.dumps({"reward": float(r[0])}, ensure_ascii=False))
        return

    if args.response_a is not None and args.response_b is not None:
        ra, rb, win = compare(model, tokenizer, args.instruction or "", args.response_a, args.response_b, args.max_length)
        print(json.dumps({"reward_a": ra, "reward_b": rb, "winner": win}, ensure_ascii=False))
        return

    raise SystemExit("Provide either --response OR both --response_a and --response_b (and --instruction).")


def run_batch(args, model, tokenizer):
    in_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    assert in_path.exists(), f"Missing input_jsonl: {in_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_prompts, batch_a, batch_b, batch_rows = [], [], [], []

    def flush():
        nonlocal batch_prompts, batch_a, batch_b, batch_rows
        if not batch_rows:
            return

        # score A and B
        rewards_a = score_texts(model, tokenizer, batch_prompts, batch_a, args.max_length)
        rewards_b = score_texts(model, tokenizer, batch_prompts, batch_b, args.max_length)

        with out_path.open("a", encoding="utf-8") as fout:
            for row, ra, rb in zip(batch_rows, rewards_a.tolist(), rewards_b.tolist()):
                winner = "A" if ra >= rb else "B"
                row_out = dict(row)
                row_out.update({"reward_a": float(ra), "reward_b": float(rb), "winner": winner})
                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")

        batch_prompts, batch_a, batch_b, batch_rows = [], [], [], []

    # Clear output file if exists
    if out_path.exists():
        out_path.unlink()

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            instr = row.get(args.instruction_key, "")
            a = row.get(args.response_a_key, "")
            b = row.get(args.response_b_key, "")

            batch_rows.append(row)
            batch_prompts.append(format_prompt(instr))
            batch_a.append(a)
            batch_b.append(b)

            if len(batch_rows) >= args.batch_size:
                flush()

    flush()
    print(f"Wrote: {out_path}")


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.base_model)

    # Batch mode
    if args.input_jsonl and args.out_jsonl:
        run_batch(args, model, tokenizer)
        return

    # Single mode
    if args.instruction is None:
        raise SystemExit("Single/compare mode requires --instruction.")
    run_single(args, model, tokenizer)


if __name__ == "__main__":
    main()

