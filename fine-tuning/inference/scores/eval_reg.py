from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned regression model on a reference-based scoring dataset.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to your fine-tuned checkpoint (or final output dir)")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to the evaluation dataset (JSONL format)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps during evaluation")
    return parser.parse_args()


max_seq_length = 2048
instruction_field = "instruction"
output_field = "output"       # single score, stored as 1..5 in JSON

# -----------------------------
# Dataset + tokenization
# -----------------------------
def get_eval_dataset():
    eval_ds = load_dataset("json", data_files={"eval": args.eval_path})["eval"]
    return eval_ds


def tokenize_and_format(example: Dict[str, Any], tokenizer: AutoTokenizer):
    text = example[instruction_field]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    score = example[output_field]
    if isinstance(score, list):
        # adapt this if you used a list and collapsed it during training
        raise ValueError(f"Expected a single score, got list: {score}")

    enc["labels"] = float(score)
    return enc



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA available, running on CPU")

    # Load tokenizer & model from your fine-tuned checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id= tokenizer.pad_token_id
    # Load and tokenize eval data
    eval_ds = get_eval_dataset()
    eval_tokenized = eval_ds.map(
        lambda ex: tokenize_and_format(ex, tokenizer),
        batched=False,
        remove_columns=eval_ds.column_names,
    )

    # Dummy training args (we won't train, just evaluate)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        logging_steps=args.logging_steps,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )


    predictions = trainer.predict(eval_tokenized)
    logits = predictions.predictions
    out_path = os.path.join(args.model_dir, "eval_predictions_1.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred_score in zip(eval_ds, logits):
            record = {
                "instruction": ex[instruction_field],
                "output":float(pred_score),          # predicted score 1–5
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved predictions to: {out_path}")
                                                     
