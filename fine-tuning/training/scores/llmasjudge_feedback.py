
"""
Finetune a causal LM to act as an LLM-as-a-judge that outputs JSON ONLY.

Key fix vs. your previous script:
- We build (prompt + target_json) but mask the loss on the prompt tokens (labels = -100),
  so the model learns to generate only the JSON part instead of echoing the prompt.

Expected JSONL fields (configurable below):
- instruction: the full evaluation prompt (task + response + reference + rubric etc.)
- output: numeric score (often float in your data); we round+clamp to int 1..5
- orig_feedback: the textual feedback to train on
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM for feedback generation.")
    parser.add_argument("--train_path", type=str, default="dataset_scores_train_feedback.jsonl", help="Path to the training dataset (JSONL format).")
    parser.add_argument("--eval_path", type=str, default="dataset_ref_based_scores_test.jsonl", help="Path to the evaluation dataset (JSONL format).")
    parser.add_argument("--output_dir", type=str, default="./judge_json_model-no-ref", help="Directory to save the fine-tuned model and tokenizer.")
    parser.add_argument("--model_name", type=str, default="google/medgemma-27b-it", help="Pre-trained model name or path.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients before updating model parameters.")
    parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Total number of training epochs to perform.")      
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduling.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log training metrics every X steps.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Run evaluation every X steps.")
    return parser.parse_args()

# ---------------- Config ----------------

@dataclass
class FinetuneConfig:

    # ---- Column names in your JSONL ----
    instruction_field: str = "instruction"
    score_field: str = "orig_score"
    feedback_field: str = "output_explanation"

    # ---- Model & tokenization ----
    max_seq_length: int = 2048

    # ---- Training ----
    save_steps: int = 200
    save_total_limit: int = 2

    # Precision
    fp16: bool = False
    bf16: bool = True 

    # Optional resume
    resume_from_checkpoint: str = None


# ---------------- Prompt/Target builders ----------------

def _score_to_int_1_5(x: Any) -> int:
    """Round + clamp to int in [1,5]."""
    try:
        v = float(x)
        v = int(round(v))
        return max(1, min(5, v))
    except Exception:
        return 3


def build_prompt_only(example: Dict[str, Any], cfg: FinetuneConfig) -> str:
    """
    Prompt ends with a hard boundary marker to reduce prompt-echo at inference.
    """
    instruction = str(example.get(cfg.instruction_field, "")).strip()

	# change prompt according to 1 score or 3 score
    prompt = (
        instruction
        + "\n\n"
          "Return JSON only. No markdown. No extra text.\n"
          #"Schema: {\"feedback\":\"...\",\"scores\":{\"Score 1\":<1-5>}, \"Score 2\":<1-5>, \"Score 3\":<1-5>}\n"
          "Schema: {\"feedback\":\"...\",\"scores\":{\"Score\":<1-5>}}\n"
	  "### JSON:\n"
    )
    return prompt


def build_target_json(example: Dict[str, Any], cfg: FinetuneConfig) -> str:
    """
    Target is strictly JSON (single object).
    """
    feedback = str(example.get(cfg.feedback_field, "")).strip()
	# change according to 1 score or 3 score
    score_int = _score_to_int_1_5(example.get(cfg.score_field, None))
    #score_int = []
    #for i in example.get(cfg.score_field,None):
	#    score_int.append(_score_to_int_1_5(i))
    obj = {"feedback": feedback, "scores": {"Score": score_int}}
    #obj = {"feedback": feedback, "scores": {"Score 1": score_int[0], "Score 2": score_int[1], "Score 2": score_int[2]}}
    # Add a newline to encourage stopping after JSON
    return json.dumps(obj, ensure_ascii=False) + "\n"


# ---------------- Preprocessing with label masking ----------------

def make_preprocess_fn(tokenizer: AutoTokenizer, cfg: FinetuneConfig):
    pad_id = tokenizer.pad_token_id

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        prompts: List[str] = []
        targets: List[str] = []

        for ins, sc, fb in zip(
            batch[cfg.instruction_field],
            batch[cfg.score_field],
            batch[cfg.feedback_field],
        ):
            ex = {
                cfg.instruction_field: ins,
                cfg.score_field: sc,
                cfg.feedback_field: fb,
            }
            prompts.append(build_prompt_only(ex, cfg))
            targets.append(build_target_json(ex, cfg))

        # Tokenize separately
        tok_p = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        tok_t = tokenizer(
            targets,
            add_special_tokens=False,  
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

        input_ids, attention_mask, labels = [], [], []

        for p_ids, t_ids in zip(tok_p["input_ids"], tok_t["input_ids"]):
            ids = p_ids + t_ids
            lab = [-100] * len(p_ids) + t_ids
            mask = [1] * len(ids)

            # Truncate
            ids = ids[: cfg.max_seq_length]
            lab = lab[: cfg.max_seq_length]
            mask = mask[: cfg.max_seq_length]

            # Pad to fixed length (Trainer is happier with fixed shapes)
            pad_len = cfg.max_seq_length - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
                lab = lab + [-100] * pad_len

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return preprocess


# ---------------- Main ----------------

def main():
    cfg = FinetuneConfig()
    args = parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else None,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data
    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    #eval_ds = load_dataset("json", data_files=args.eval_path, split="train")

    preprocess_fn = make_preprocess_fn(tokenizer, cfg)
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    #eval_ds = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    # Training args
    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        #eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to="none",
        metric_for_best_model="spearman",
        remove_unused_columns=False, 
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        #eval_dataset=eval_ds,
        tokenizer=tokenizer,
        #processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

