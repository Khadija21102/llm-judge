from dataclasses import dataclass
from typing import Dict, Any, List
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW


@dataclass
class FinetuneConfig:
    # --- paths ---
    train_path: str = "dataset_pref_train_clean.jsonl"
    eval_path: str = "dataset_pref_test.jsonl"

    # --- fields in your JSONL ---
    question_field: str = "orig_instruction"
    response_A_field: str = "orig_response_A"
    response_B_field: str = "orig_response_B"
    winner_field: str = "output"  # "A" or "B"

    # --- model ---
    model_name: str = "google/medgemma-4b-it" 
    max_seq_length: int = 2048

    # --- training ---
    output_dir: str = "./pairwise-reward-medgemma-no-ref"
    train_batch_size: int = 8
    eval_batch_size: int = 1
    learning_rate: float = 1e-6
    num_train_epochs: int = 5
    weight_decay: float = 0.0

    seed: int = 42


cfg = FinetuneConfig()


def build_single_prompt(example: Dict[str, Any], which: str) -> str:
    """Prompt for ONE candidate response (A or B)."""
    question = example.get(cfg.question_field, "")

    if which == "A":
        candidate = example[cfg.response_A_field]
    else:
        candidate = example[cfg.response_B_field]

    prompt = f"""
You are an expert medical judge. You will evaluate ONE candidate response to a clinical question.

### Question
{question}

### Candidate Response
{candidate}

Your task is to internally decide how good this response is (harmlessness, alignment with guidelines, relevance, completeness).
You do NOT need to output the score; the model will learn an internal scalar quality score.
""".strip()
    return prompt


def preprocess_pairwise(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Build tokenized inputs for a pair (A, B) + winner.
    winner: +1 if first is better, -1 if second is better (after random swap).
    """
    winner_str = example[cfg.winner_field]  # "A" or "B"
    winner = 1 if winner_str == "A" else -1

    prompt_A = build_single_prompt(example, "A")
    prompt_B = build_single_prompt(example, "B")

    # Randomly swap order to avoid positional bias
    if random.random() < 0.5:
        prompt_A, prompt_B = prompt_B, prompt_A
        winner = -winner  # if swapped, invert

    enc_A = tokenizer(
        prompt_A,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )
    enc_B = tokenizer(
        prompt_B,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
    )

    return {
        "input_ids_A": enc_A["input_ids"],
        "attention_mask_A": enc_A["attention_mask"],
        "input_ids_B": enc_B["input_ids"],
        "attention_mask_B": enc_B["attention_mask"],
        "winner": winner,   # +1 if A (first) better, -1 if B (second) better
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simple collator: stack already-padded lists into tensors.
    """
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [ex[k] for ex in batch]
        if isinstance(vals[0], list):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return out


def train_and_eval():
    # --- Seeding ---
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # --- Load dataset ---
    raw_datasets = load_dataset(
        "json",
        data_files={"train": cfg.train_path, "eval": cfg.eval_path},
    )

    from collections import Counter
    print("Train winners (raw):", Counter(raw_datasets["train"][cfg.winner_field]))
    print("Eval  winners (raw):", Counter(raw_datasets["eval"][cfg.winner_field]))

    # --- Tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1,  # scalar score
        device_map="auto",
        torch_dtype=torch.bfloat16,   
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    model.config.problem_type = "regression"
    model.config.pad_token_id = tokenizer.pad_token_id

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # --- Preprocess datasets to pairwise format ---
    def _preprocess(ex):
        return preprocess_pairwise(ex, tokenizer)

    train_ds = raw_datasets["train"].map(_preprocess, batched=False)
    eval_ds = raw_datasets["eval"].map(_preprocess, batched=False)

    # Remove original fields to keep only tensors we need
    keep_cols = ["input_ids_A", "attention_mask_A", "input_ids_B", "attention_mask_B", "winner"]
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in keep_cols]
    )
    eval_ds = eval_ds.remove_columns(
        [c for c in eval_ds.column_names if c not in keep_cols]
    )

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # --- Training loop ---
    for epoch in range(cfg.num_train_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            bs = batch["input_ids_A"].size(0)

            # Concatenate A and B for a single forward
            input_ids = torch.cat([batch["input_ids_A"], batch["input_ids_B"]], dim=0)
            attention_mask = torch.cat([batch["attention_mask_A"], batch["attention_mask_B"]], dim=0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits.view(-1)  # (2*bs,)

            s_A = scores[:bs]
            s_B = scores[bs:]

            # winner: +1 if A (first) better, -1 if B (second) better
            margin = batch["winner"].float() * (s_A - s_B)  # (bs,)
            loss = -F.logsigmoid(margin).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(1, n_steps)
        print(f"Epoch {epoch+1}/{cfg.num_train_epochs} - train loss: {avg_loss:.4f}")

        # --- Eval after each epoch ---
        model.eval()
        total_correct = 0
        total_pairs = 0

        with torch.no_grad():
            for batch in eval_loader:
                #batch = {k: v.to(device) for k, v in batch.items()}
                bs = batch["input_ids_A"].size(0)

                input_ids = torch.cat([batch["input_ids_A"], batch["input_ids_B"]], dim=0)
                attention_mask = torch.cat([batch["attention_mask_A"], batch["attention_mask_B"]], dim=0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                scores = outputs.logits.view(-1)

                s_A = scores[:bs]
                s_B = scores[bs:]
                logits_device = scores.device
                pred_winner = torch.where(
                    s_A > s_B,
                    torch.tensor(1, device=logits_device),
                    torch.tensor(-1, device=logits_device),
                )
                # model's choice:  +1 if A better, -1 if B better
                #pred_winner = torch.where(s_A > s_B, torch.tensor(1, device=device), torch.tensor(-1, device=device))

                # compare with true winner
                correct = (pred_winner == batch["winner"]).sum().item()
                total_correct += correct
                total_pairs += bs

        acc = total_correct / max(1, total_pairs)
        print(f"Epoch {epoch+1} - eval pairwise accuracy: {acc:.4f}")

    # Optionally save model
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Model saved to", cfg.output_dir)


if __name__ == "__main__":
    train_and_eval()

