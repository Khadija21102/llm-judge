#!/usr/bin/env python3
"""
Full fine-tuning (no LoRA) to generate:
### Answer:
...
### Feedback:
...

Input JSONL fields (defaults match your dataset):
- instruction
- orig_response_A
- orig_response_B
- output: "A" or "B"
- explanation: feedback text

Train text format:
### Instruction:
...
### Output:
### Answer:
...
### Feedback:
...

Run with :
  python reward_model_feedback.py \
    --train_jsonl /path/to/dataset_ref_based_pref_train_clean_with_explanations.jsonl \
    --model_id google/medgemma-27b-text-it \
    --output_dir ./fullft-medgemma-answer-feedback \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
"""

import argparse
import math
from typing import Dict

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from trl import RewardTrainer, RewardConfig
from utils.clean import clean_jsonl
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=False, default="dataset_ref_based_pref_train_new_v2_with_explanations_clean.jsonl")
    p.add_argument("--eval_jsonl", type=str, default=None)

    p.add_argument("--model_id", type=str, required=False, default="google/medgemma-27b-it")
    p.add_argument("--output_dir", type=str, required=False, default="pref-reward-feedback-v3")

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Dataset keys
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--resp_a_key", type=str, default="orig_response_A")
    p.add_argument("--resp_b_key", type=str, default="orig_response_B")
    p.add_argument("--label_key", type=str, default="output")  # "A"/"B" or 0/1
    p.add_argument("--explanation_key", type=str, default="explanation")

    return p.parse_args()


def normalize_label(x):
    if x is None:
        return None
    if isinstance(x, str):
        v = x.strip().upper()
        if v in {"A", "B"}:
            return v
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        if int(x) == 0:
            return "A"
        if int(x) == 1:
            return "B"
    return None


def build_text(example: Dict, args) -> Dict:
    instr = (example.get(args.instruction_key, "") or "").strip()
    a = (example.get(args.resp_a_key, "") or "").strip()
    b = (example.get(args.resp_b_key, "") or "").strip()
    lab = normalize_label(example.get(args.label_key))
    expl = (example.get(args.explanation_key, "") or "").strip()

    if lab == "A":
        answer = a
    elif lab == "B":
        answer = b
    else:
        return {"text": None}

    if not answer or not expl:
        return {"text": None}

    text = (
        "You are a helpful assistant. Follow the instruction and provide an answer and feedback.\n\n"
        f"### Instruction:\n{instr}\n\n"
        "### Output:\n"
        f"### Answer:\n{answer}\n\n"
        f"### Feedback:\n{expl}\n"
    )
    return {"text": text}


def main():
    args = parse_args()
    # Clean the data (removes duplicates and bad formatting)
    clean_jsonl(args.train_jsonl, args.train_jsonl)
    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl:
        data_files["eval"] = args.eval_jsonl

    ds = load_dataset("json", data_files=data_files)

    ds = ds.map(lambda ex: build_text(ex, args))
    ds = ds.filter(lambda ex: ex["text"] is not None)

    train_ds = ds["train"]
    eval_ds = ds["eval"] if "eval" in ds else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Full-precision model load for full fine-tuning (no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
    )
    model.config.use_cache = False  # important for training

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        bf16=True,        
        fp16=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        #dataset_text_field="text",
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved full fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()

