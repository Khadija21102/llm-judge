from dataclasses import dataclass
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from typing import Any, Dict, Tuple, Optional

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for regression on score data.")
    parser.add_argument("--model_name", type=str, default="google/medgemma-27b-it", help="Pre-trained model name or path.")
    parser.add_argument("--train_path", type=str, default="dataset_ref_based_scores_train_balanced.jsonl", help="Path to training dataset (JSONL).")
    parser.add_argument("--eval_path", type=str, default="dataset_ref_based_scores_test.jsonl", help="Path to evaluation dataset (JSONL).")
    parser.add_argument("--output_dir", type=str, default="./medgemma-27b-reg-v3", help="Directory to save the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--num_train_epochs", type=float, default=8.0, help="Total number of training epochs (can be a float for fractional epochs).")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Number of steps between logging training metrics.")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving model checkpoints.")
    parser.add_argument("--use_4bit", action='store_true', help="Whether to use 4-bit quantization (requires bitsandbytes).")
    parser.add_argument("--use_lora", action='store_true', help="Whether to use LoRA for parameter-efficient fine-tuning.")

    args = parser.parse_args()
    return args
@dataclass
class FinetuneConfig:

    instruction_field: str = "instruction"
    output_field: str = "output"          # <-- single float score now

    max_seq_length: int = 2048


cfg = FinetuneConfig()


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

    # Expect a single scalar (int/float). If it's a list, take the first or raise.
    if isinstance(score, list):
        # You can choose to take the first element instead:
        # score = score[0]
        raise ValueError(f"Expected a single score, got list: {score}")

    enc["labels"] = float(score)  # HF will make this a float tensor
    return enc


if __name__ == "__main__":
    # Safer CUDA print
    args = parse_args()
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
        num_labels=1,                
        problem_type="regression",
        torch_dtype=torch.bfloat16 if use_bf16 and not args.use_4bit else torch.float32,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        token=hf_token,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
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
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,                # only true on GPU
        fp16=False,                   # you can flip this if you actually want fp16
        eval_steps=args.save_steps if eval_strategy == "steps" else None,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        # Uncomment if your transformers version supports it:
        # evaluation_strategy=eval_strategy,
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

