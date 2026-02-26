#!/usr/bin/env python3
"""
Train a reward model (pairwise preference) using MedGemma 27B as base model.

- Loads JSONL with fields:
    instruction, orig_response_A, orig_response_B, output (A/B)
- Converts to TRL format: prompt, chosen, rejected
- Trains with TRL RewardTrainer
Example:
  python train_reward_model_medgemma27b.py \
    --train_jsonl /path/to/dataset_ref_based_pref_train_clean_with_explanations.jsonl \
    --model_id google/medgemma-27b-text-it \
    --output_dir ./rm-medgemma27b \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 1
"""

import argparse
import os
import re
import math
from typing import Dict

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from trl import RewardTrainer, RewardConfig
from utils.clean import clean_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=False, help="Path to training JSONL.", default="dataset_ref_based_pref_train_clean.jsonl")
    p.add_argument("--eval_jsonl", type=str, default=None, help="Optional eval JSONL path.")
    p.add_argument("--model_id", type=str, default="google/medgemma-27b-it",
                   help="HF model id (e.g., google/medgemma-27b-text-it or google/medgemma-27b-it).")
    p.add_argument("--output_dir", type=str, required=False, default= "pref-reward-model-v11")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Data field names (defaults match your file)
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--resp_a_key", type=str, default="orig_response_A")
    p.add_argument("--resp_b_key", type=str, default="orig_response_B")
    p.add_argument("--label_key", type=str, default="output")  # "A" or "B"

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


def format_prompt(instruction: str) -> str:
    """
    Simple prompt format for reward modeling.
    Keep prompt separate from responses; RewardTrainer will pair with chosen/rejected.
    """
    instruction = (instruction or "").strip()
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def build_pairwise(example: Dict, args) -> Dict:
    instr = example.get(args.instruction_key, "")
    a = example.get(args.resp_a_key, "")
    b = example.get(args.resp_b_key, "")
    lab = normalize_label(example.get(args.label_key))

    prompt = format_prompt(instr)

    if lab == "A":
        chosen = (a or "").strip()
        rejected = (b or "").strip()
    elif lab == "B":
        chosen = (b or "").strip()
        rejected = (a or "").strip()
    else:
        # drop unusable examples
        return {"prompt": None, "chosen": None, "rejected": None}

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Clean the data (removes duplicates and bad formatting)
    clean_jsonl(args.train_jsonl, args.train_jsonl)
    # Load dataset(s)
    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl:
        data_files["eval"] = args.eval_jsonl

    ds = load_dataset("json", data_files=data_files)

    ds = ds.map(lambda ex: build_pairwise(ex, args), remove_columns=ds["train"].column_names)
    ds = ds.filter(lambda ex: ex["prompt"] is not None and ex["chosen"] and ex["rejected"])

    train_ds = ds["train"]
    eval_ds = ds["eval"] if "eval" in ds else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit quantization (QLoRA)
    #bnb_config = BitsAndBytesConfig(
     #   load_in_4bit=True,
      #  bnb_4bit_quant_type="nf4",
       # bnb_4bit_compute_dtype=torch.bfloat16,
        #bnb_4bit_use_double_quant=True,
    #)

    # Reward model head: num_labels=1 (scalar reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        #quantization_config=bnb_config,
        device_map="auto",
    )

    # Enable gradient checkpointing for memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # important for training

    # LoRA adapters (common target modules for Gemma-like architectures)
    # If you hit shape/key errors, adjust target_modules.
    #lora_config = LoraConfig(
     #   r=args.lora_r,
      #  lora_alpha=args.lora_alpha,
       # lora_dropout=args.lora_dropout,
        #bias="none",
        #task_type=TaskType.SEQ_CLS,
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #)
    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()

    # TRL RewardConfig (TrainingArguments wrapper)
    reward_args = RewardConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
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

    trainer = RewardTrainer(
        model=model,
        args=reward_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        #tokenizer=tokenizer,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save adapter + tokenizer (recommended with QLoRA)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved reward model adapters + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()

