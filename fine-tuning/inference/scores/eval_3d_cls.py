#!/usr/bin/env python
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on 3D classification task.")
    parser.add_argument("--model_name", type=str, default="judge_json_model-no-ref-3d", help="Name of the fine-tuned model (directory).")
    parser.add_argument("--eval_path", type=str, default="dataset_ref_based_scores_test_new_v2_3d.jsonl", help="Path to evaluation JSONL file.")
    parser.add_argument("--output_dir", type=str, default="./medgemma-3task-cls-no-ref", help="Directory where the fine-tuned model is saved.")
    parser.add_argument("--predictions_path", type=str, default="predictions_eval.jsonl", help="Path to save predictions JSONL file.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation.")
    return parser.parse_args()
# -----------------------------
# Config
# -----------------------------
@dataclass
class FinetuneConfig:
    # Fields in JSONL
    instruction_field: str = "instruction"
    output_field: str = "orig_score"          # list of 3 integers (1–5)

    # Tokenization / batching
    max_seq_length: int = 2048
    num_tasks: int = 3
    num_classes: int = 5

cfg = FinetuneConfig()

args = parse_args()
# -----------------------------
# Tokenization
# -----------------------------
def tokenize_and_format(example: Dict[str, Any], tokenizer: AutoTokenizer):
    text = example[cfg.instruction_field]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )

    scores = example.get(cfg.output_field, None)
    if scores is not None:
        if not isinstance(scores, list) or len(scores) != cfg.num_tasks:
            raise ValueError(f"Expected list of {cfg.num_tasks} scores, got: {scores}")

        labels = []
        for s in scores:
            s_int = int(round(float(s)))
            if s_int < 1:
                s_int = 1
            elif s_int > cfg.num_classes:
                s_int = cfg.num_classes
            labels.append(s_int - 1)  # shift 1–5 -> 0–4

        enc["labels"] = labels 

    return enc


# -----------------------------
# Model definition (same as training)
# -----------------------------
class MultiTaskMistralClassifier(PreTrainedModel):
    """
    Same wrapper you used for training:
      - base: Mistral (via AutoModel)
      - head: linear -> (num_tasks x num_classes) logits
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

    # ---- Gradient checkpointing support (optional here, but harmless) ----
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = True

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

        # pool last non-pad token
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            lengths = lengths.to(hidden_states.device).long()
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
# Main eval script
# -----------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        print("No CUDA available, running on CPU")
        device_map = None
        torch_dtype = torch.float32

    # Load tokenizer (from fine-tuned dir if available)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load raw eval dataset
    eval_ds = load_dataset("json", data_files={"eval": args.eval_path})["eval"]

    # Tokenize eval dataset
    eval_tokenized = eval_ds.map(
        lambda ex: tokenize_and_format(ex, tokenizer),
        batched=False,
        remove_columns=eval_ds.column_names,
    )

    # Load config & model from fine-tuned directory
    config = AutoConfig.from_pretrained(args.output_dir)
    # ensure task info is there (if not already saved)
    config.num_classes = getattr(config, "num_classes", cfg.num_classes)
    config.num_tasks = getattr(config, "num_tasks", cfg.num_tasks)

    model = MultiTaskMistralClassifier.from_pretrained(
        args.output_dir,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Eval-only TrainingArguments
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_runs"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )


    pred_output = trainer.predict(eval_tokenized)
    pred_logits = pred_output.predictions  # (N, 3, 5)
    pred_classes = pred_logits.argmax(axis=-1)  # (N, 3) in [0..4]
    pred_scores = pred_classes + 1             # back to [1..5]

    # 3) Write file with prompt + model output
    print(f"💾 Writing predictions to {args.predictions_path} ...")
    with open(args.predictions_path, "w", encoding="utf-8") as f:
        for i in range(len(eval_ds)):
            record = {
                "instruction": eval_ds[i][cfg.instruction_field],
                "gold_scores": eval_ds[i].get(cfg.output_field, None),
                "pred_scores": pred_scores[i].tolist(),  # [s1, s2, s3] in 1–5
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("✅ Done.")
