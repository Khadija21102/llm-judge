"""
Fine-tuning script for models on the Clinicians-LLM dataset, using a multi-task classification head to predict 3 separate scores from 1 to 5. The script loads pre-split datasets, tokenizes them, and trains the model using Hugging Face's Trainer API, saving the best model to the specified output directory.
"""
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from transformers import PreTrainedModel, AutoModel, AutoConfig

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model on the Clinicians-LLM dataset for multi-task classification.")
    parser.add_argument("--model_name", type=str, default="google/medgemma-27b-it", help="Pre-trained model name or path")
    parser.add_argument("--train_path", type=str, default="dataset_scores_train_3d.jsonl", help="Path to training dataset (JSONL)")
    parser.add_argument("--eval_path", type=str, default="dataset_scores_test_3d.jsonl", help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./medgemma-3task-cls-no-ref", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log training metrics every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model checkpoint every X steps")
    return parser.parse_args()
# -----------------------------
# Config
# -----------------------------
@dataclass
class FinetuneConfig:

    instruction_field: str = "instruction"
    output_field: str = "orig_score"          # list of 3 integers (1–5)

    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_steps: int = 500

    num_tasks: int = 3
    num_classes: int = 5


cfg = FinetuneConfig()
args = parse_args()

# -----------------------------
# Dataset loading
# -----------------------------
def get_datasets():
    train_ds = load_dataset("json", data_files={"train": args.train_path})["train"]

    eval_ds = None
    if args.eval_path is not None:
        eval_ds = load_dataset("json", data_files={"eval": args.eval_path})["eval"]

    return train_ds, eval_ds


# -----------------------------
# Tokenization + label formatting
# -----------------------------
def tokenize_and_format(example: Dict[str, Any], tokenizer: AutoTokenizer):
    text = example[cfg.instruction_field]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )

    scores = example[cfg.output_field]   # expected like [5, 4, 3] or [5.0,4.0,3.0]
    if not isinstance(scores, list) or len(scores) != cfg.num_tasks:
        raise ValueError(f"Expected list of {cfg.num_tasks} scores, got: {scores}")

    # Convert to ints 1–5, then to 0–4 for classification
    labels = []
    for s in scores:
        s_int = int(round(float(s)))
        if s_int < 1:
            s_int = 1
        elif s_int > cfg.num_classes:
            s_int=5
        labels.append(s_int - 1)  # shift to 0–4

    enc["labels"] = labels  # shape (3,)
    return enc




class MultiTaskMistralClassifier(PreTrainedModel):
    """
    PreTrainedModel wrapper:
      - base: Mistral (via AutoModel)
      - head: linear -> (num_tasks x num_classes) logits
      - labels: (batch, num_tasks) with values in [0, num_classes-1]
    """

    config_class = AutoConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        # task info from config
        self.num_tasks = getattr(config, "num_tasks", 3)
        self.num_classes = getattr(config, "num_classes", 5)

        self.model = AutoModel.from_config(config)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(config, "dim", None)
        if hidden_size is None:
            # last resort: read from embeddings
            emb = self.model.get_input_embeddings()
            if emb is None or not hasattr(emb, "weight"):
                raise ValueError("Could not infer hidden size: no config field and no input embeddings.")
            hidden_size = emb.weight.shape[1]

        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(
            hidden_size, self.num_tasks * self.num_classes
        )

        self.pad_token_id = getattr(config, "pad_token_id", None)

        self.post_init()

    # ---- Gradient checkpointing support (override directly) ----
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Called by Trainer when gradient_checkpointing=True.
        We simply delegate to the underlying base model if it supports it.
        """
        if hasattr(self.model, "gradient_checkpointing_enable"):
            # some models accept kwargs, some don't
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, "gradient_checkpointing"):
            # older style flag
            self.model.gradient_checkpointing = True

        # mark in config (helps some HF utilities)
        if hasattr(self.config, "gradient_checkpointing"):
            self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        elif hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = False

        if hasattr(self.config, "gradient_checkpointing"):
            self.config.gradient_checkpointing = False

    # ------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            lengths = lengths.to(hidden_states.device).long()  # <-- key fix
            batch_indices = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            pooled = hidden_states[batch_indices, lengths]
        else:
            pooled = hidden_states[:, -1, :]


        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)  # (batch, num_tasks * num_classes)
        logits = logits.view(-1, self.num_tasks, self.num_classes)  # (B, 3, 5)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).long()  # (B, 3)
            loss_fct = nn.CrossEntropyLoss()
            loss = 0.0
            for t in range(self.num_tasks):
                loss = loss + loss_fct(logits[:, t, :], labels[:, t])
            loss = loss / self.num_tasks

        return {"loss": loss, "logits": logits}




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

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load raw datasets
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

    # Build model

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_classes= cfg.num_classes
    config.num_tasks= cfg.num_tasks
    model = MultiTaskMistralClassifier.from_pretrained(
	    args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        )

    # Training args
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
        eval_steps=args.save_steps if eval_tokenized is not None else None,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Finished training. Saved to", args.output_dir)

