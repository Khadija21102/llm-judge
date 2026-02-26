#!/usr/bin/env python3
"""
Inference script to evaluate a fine-tuned LLM on a dataset with 3 regression scores. Saves the instruction, gold scores, and predicted scores to a new JSONL file.
"""
from dataclasses import dataclass
import os
from typing import Any, Dict

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
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned regression model on a reference-based scoring dataset with 3 scores.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to your fine-tuned checkpoint (or final output dir)")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to the evaluation dataset (JSONL format)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps during evaluation")
    return parser.parse_args()

# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:

    instruction_field: str = "instruction"
    output_field: str = "output"       # list of 3 floats

    max_seq_length: int = 2048

    num_outputs: int = 3               # 3 regression scores
    save_pred_file: str = "eval_predictions_3d_regression.jsonl"


cfg = EvalConfig()

args = parse_args()
# -----------------------------
# Dataset + tokenization
# -----------------------------
def get_eval_dataset():
    eval_ds = load_dataset("json", data_files={"eval": args.eval_path})["eval"]
    return eval_ds


def tokenize_and_format(example: Dict[str, Any], tokenizer: AutoTokenizer):
    text = example[cfg.instruction_field]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )

    scores = example[cfg.output_field]   # expected list of 3 floats
    if not isinstance(scores, list) or len(scores) != cfg.num_outputs:
        raise ValueError(f"Expected list of {cfg.num_outputs} scores, got: {scores}")

    # store as-is (floats) for regression
    enc["labels"] = [float(s) for s in scores]

    return enc


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        device_map = "auto"
    else:
        print("No CUDA available, running on CPU")
        device_map = None

    # Load tokenizer & model from your fine-tuned checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=cfg.num_outputs,
        problem_type="regression",
        torch_dtype=torch.float32,
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load and tokenize eval data
    eval_ds = get_eval_dataset()
    eval_tokenized = eval_ds.map(
        lambda ex: tokenize_and_format(ex, tokenizer),
        batched=False,
        remove_columns=eval_ds.column_names,
    )

    # Evaluation-only args
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


    # ----- 1) Run prediction and save instruction + gold + predicted 3 scores -----
    predictions = trainer.predict(eval_tokenized)
    preds = np.array(predictions.predictions)  # (N, 3) or (N, 3, 1)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)

    preds_list = preds.tolist()  # make JSON-serializable

    out_path = os.path.join(args.model_dir, cfg.save_pred_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred_scores in zip(eval_ds, preds_list):
            record = {
                "instruction": ex[cfg.instruction_field],
                "gold_output": ex[cfg.output_field],          # original 3 scores
                "pred_scores": [float(x) for x in pred_scores],  # predicted 3 scores
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved predictions to: {out_path}")

