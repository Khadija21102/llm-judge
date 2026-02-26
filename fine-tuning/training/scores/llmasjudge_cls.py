"""
Fine-tuning script for LLaMA-based models on the Clinicians-LLM dataset, using a classification head to predict scores from 1 to 5. The script supports optional 4-bit quantization and LoRA fine-tuning for efficient training. It loads pre-split datasets, tokenizes them, and trains the model using Hugging Face's Trainer API, saving the best model to the specified output directory.

"""

from dataclasses import dataclass
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch.nn as nn

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA-based model on the Clinicians-LLM dataset.")
    parser.add_argument("--model_name", type=str, default="google/medgemma-27b-it", help="Pre-trained model name or path.")
    parser.add_argument("--train_path", type=str, default="dataset_scores_train_new.jsonl", help="Path to the training dataset (JSONL format).")
    parser.add_argument("--eval_path", type=str, default="dataset_scores_test.jsonl", help="Path to the evaluation dataset (JSONL format).")
    parser.add_argument("--output_dir", type=str, default="./medgemma-no-ref", help="Directory to save the fine-tuned model and tokenizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating model parameters.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for the optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Number of steps between logging training metrics.")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving model checkpoints.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--use_4bit", action='store_true', help="Whether to use 4-bit quantization for the model.")
    parser.add_argument("--use_lora", action='store_true', help="Whether to use LoRA fine-tuning for the model.")
    return parser.parse_args()

@dataclass
class FinetuneConfig:

    instruction_field: str = "instruction"
    output_field: str = "output"       
    max_seq_length: int = 2048


    num_classes:int = 5
cfg = FinetuneConfig()
   
args = parse_args()
def get_datasets() -> Tuple[Any, Optional[Any]]:
    """Load pre-split datasets."""
    train_ds = load_dataset("json", data_files={"train": args.train_path})["train"]

    eval_ds = None
    if args.eval_path is not None:
        eval_ds = load_dataset("json", data_files={"eval": args.eval_path})["eval"]

    return train_ds, eval_ds


def tokenize_and_format(example: Dict[str, Any], tokenizer: AutoTokenizer):
    text = example[cfg.instruction_field]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )

    score = example[cfg.output_field]
    if not np.isnan(score):
        score = int(round(float(score)))
    else:
        print("Nan")
    if score < 1:
        score=1
    if score > cfg.num_classes:
        score=5
    enc["labels"] = score-1
    return enc

if __name__ == "__main__":
    # Safer CUDA print
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA available, running on CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", None)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load your pre-split data
    train_ds, eval_ds = get_datasets()

    # Tokenize
    train_tokenized = train_ds.map(
        lambda ex: tokenize_and_format(ex, tokenizer),
        batched=False,
        remove_columns=train_ds.column_names,
    )

    eval_tokenized = None
    if eval_ds is not None:
        eval_tokenized = eval_ds.map(
            lambda ex: tokenize_and_format(ex, tokenizer),
            batched=False,
            remove_columns=eval_ds.column_names,
        )
    
    # 1-d regression head
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # dtype: bf16 only if CUDA is available, otherwise float32
    use_bf16 = torch.cuda.is_available()

    model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=5,
    problem_type="single_label_classification",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Eval strategy
    eval_strategy = "steps" if eval_tokenized is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay= args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_steps=args.save_steps if eval_strategy == "steps" else None,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Finished training. Saved to", args.output_dir)
