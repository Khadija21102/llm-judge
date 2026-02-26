#!/usr/bin/env python3
"""
===============================================================================
LLM vs Clinician Preference Evaluation (Unified Version)
===============================================================================

This script evaluates agreement between clinician A/B preferences and an LLM's
preference decision.

Supported models via --model:
  - fine-tuning   (built-in extraction; no external parser needed)
  - meditron
  - prometheus
  - gpt
  - claude
  - llama
  - jury (placeholder; handle later)

External parser requirement:
  For models other than fine-tuning, this script imports a module named:
      parse_<model>.py
  which must define:
      parse_<model>(record) -> (response_A, response_B, llm_result)
  where llm_result is one of: 'A', 'B', 'Tie', or None.

Score-field:
    
Clinician CSV:
  Must contain A/B texts and a preference indicator. Two supported schemas:
    1) "Vote" (CHUV format): Vote==1 -> A else -> B
       with columns: "First Answer", "Second Answer"
    2) "First Answer Improved": notna -> A else -> B
       with columns: "First Answer", "Second Answer"

Clinician preference per pair is aggregated by majority vote: A/B/Tie.

Metrics (--metric):
  - accuracy : exact match rate between clinician_result and llm_result (ties removed)
  - kappa    : Cohen's kappa (ties removed)
  - icc      : ICC3 on mapped ratings A->1, B->0 (ties removed)

Examples:
  Fine-tuning:
    python new_pers-clinicians-llm-pref.py --model fine-tuning --llm_jsonl inference.jsonl \
      --clinician_csv df_evaluation_pref.csv --metric icc

  Meditron:
    python new_pers-clinicians-llm-pref.py --model meditron --llm_jsonl meditron_pref.jsonl \
      --clinician_csv df_evaluation_pref.csv --metric kappa_w

  GPT (needs ref_jsonl for response_A/B):
    python new_pers-clinicians-llm-pref.py --model gpt --llm_jsonl gpt_pref.jsonl \
      --ref_jsonl dataset_ref_based_pref_test.jsonl --clinician_csv df_evaluation_pref.csv --metric accuracy
===============================================================================
"""

import argparse
import importlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score


# ---------------------------
# Utilities
# ---------------------------
def load_external_parser(model_name: str):
    module_name = f"parse_{model_name}_pref"
    func_name = f"parse_{model_name}_pref"
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise ImportError(f"Module '{module_name}.py' does not define '{func_name}(record)'.")
    return fn


def normalize_ab(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip().upper()
        if x == "TIE":
            return "Tie"
        if x in {"A", "B", "Tie"}:
            return x
    return None


def majority_vote(results: list) -> Optional[str]:
    """Majority vote over ['A','B'] clinician results; returns 'A', 'B', or 'Tie'."""
    if not results:
        return None
    count_a = sum(1 for r in results if r == "A")
    count_b = sum(1 for r in results if r == "B")
    if count_a > count_b:
        return "A"
    if count_b > count_a:
        return "B"
    return "Tie"


def load_ref_df(ref_jsonl: Optional[str]) -> Optional[pd.DataFrame]:
    if not ref_jsonl:
        return None
    return pd.read_json(ref_jsonl, lines=True)


def drop_ties(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[df[col].isin(["A", "B"])].copy()


# ---------------------------
# Build LLM preference DF
# ---------------------------
def build_llm_df_finetuning(llm_jsonl: str, score_field: str) -> pd.DataFrame:
    """
    Fine-tuning preference outputs typically contain:
      - orig_response_A
      - orig_response_B
      - winner  (or preference / predicted_label)
    Mirrors your pers-clinicians-llm-pref.py logic. :contentReference[oaicite:1]{index=1}
    """
    rows = []
    failures = 0

    with open(llm_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)

            ra = data.get("orig_response_A")
            rb = data.get("orig_response_B")

            try:
                res = data.get(score_field)
            except Exception as e:
                print(f"Error extracting score_field '{score_field}' from record: {e}")
                res = None
            

            res = normalize_ab(res)

            if res is None or not isinstance(ra, str) or not isinstance(rb, str):
                failures += 1
                continue

            rows.append({"response_A": ra.strip(), "response_B": rb.strip(), "llm_result": res})

    df_llm = pd.DataFrame(rows)
    print(f"(fine-tuning) parsing failures: {failures} | kept: {len(df_llm)}")
    return df_llm


def build_llm_df_external(
    llm_jsonl: str,
    parse_fn,
    model: str,
    ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For external parsers parse_<model>(record)->(response_A,response_B,llm_result).
    If response_A/B are None and ref_df is provided, fills them from ref_df by index.
    """
    rows = []
    failures = 0

    ref_df = pd.read_csv(ref)
    with open(llm_jsonl, "r") as f:
        for (i, row), line in zip(ref_df.iterrows(), f):
            data = json.loads(line)
            try:
                if model == "meditron" or model == "prometheus":
                    ra, rb, res = parse_fn(data)
                    print(f"Parsed (meditron/prometheus) - response_A: {ra}, response_B: {rb}, llm_result: {res}")
                else:
                    res = parse_fn(data)
                    res = normalize_ab(res)
                    ra, rb = row["First Answer"], row["Second Answer"]
            except Exception:
                ra, rb, res = None, None, None


            if res is None or not isinstance(ra, str) or not isinstance(rb, str):
                failures += 1
                continue

            rows.append({"response_A": ra, "response_B": rb, "llm_result": res})

    df_llm = pd.DataFrame(rows)
    print(f"(external) parsing failures: {failures} | kept: {len(df_llm)}")
    return df_llm


# ---------------------------
# Build clinician preference DF
# ---------------------------
def build_clinician_df_pref(clinician_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clinician_csv)

    if "Vote" in df.columns:
        df["clinician_result"] = df["Vote"].apply(lambda x: "A" if x == 1 else "B")
        first_col, second_col = "First Answer", "Second Answer"
    elif "First Answer Improved" in df.columns:
        df["clinician_result"] = df["First Answer Improved"].apply(lambda x: "A" if pd.notna(x) else "B")
        first_col, second_col = "First Answer", "Second Answer"
    else:
        raise ValueError(
            "Clinician CSV must contain either 'Vote' or 'First Answer Improved', "
            "and must include 'First Answer' and 'Second Answer' columns."
        )

    df_pairs = (
        df.groupby([first_col, second_col], as_index=False)["clinician_result"]
          .agg(lambda s: majority_vote(list(s)))
          .rename(columns={first_col: "response_A", second_col: "response_B"})
    )
    return df_pairs


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["fine-tuning", "meditron", "prometheus", "gpt", "claude", "llama", "jury"])
    parser.add_argument("--llm_jsonl", type=str, required=True)
    parser.add_argument("--clinician_csv", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True,
                        choices=["accuracy", "kappa", "icc"])
    parser.add_argument("--score_field", type=str, required=True, help="Field name for direct score extraction")
    args = parser.parse_args()

    # LLM preferences
    if args.model == "fine-tuning":
        df_llm = build_llm_df_finetuning(args.llm_jsonl)
    else:
        parse_fn = load_external_parser(args.model)
        df_llm = build_llm_df_external(
            llm_jsonl=args.llm_jsonl,
            parse_fn=parse_fn,
            model=args.model,
            ref=args.clinician_csv, 
        )

    # Clinicians
    df_clin = build_clinician_df_pref(args.clinician_csv)

    # Merge on response texts
    merged = pd.merge(df_clin, df_llm, on=["response_A", "response_B"], how="inner")
    merged = merged.dropna(subset=["clinician_result", "llm_result"])

    # Drop ties for A/B metrics
    merged = drop_ties(merged, "clinician_result")
    merged = drop_ties(merged, "llm_result")

    print("Merged rows:", len(merged))
    if len(merged) == 0:
        print("Empty merge. Check that response_A/response_B match between clinician and LLM sources.")
        return

    if args.metric == "accuracy":
        acc = float((merged["clinician_result"] == merged["llm_result"]).mean())
        print(f"Accuracy: {acc:.6f}")

    elif args.metric == "kappa":
        k = cohen_kappa_score(merged["clinician_result"], merged["llm_result"])
        print(f"Cohen kappa: {k:.6f}")

    elif args.metric == "icc":
        mapping = {"A": 1, "B": 0}
        df_icc = pd.melt(
            merged,
            id_vars=["response_A", "response_B"],
            value_vars=["clinician_result", "llm_result"],
            var_name="rater",
            value_name="rating",
        )
        df_icc["targets"] = df_icc["response_A"] + " ||| " + df_icc["response_B"]
        df_icc["rating"] = df_icc["rating"].map(mapping).astype(int)

        icc_results = pg.intraclass_corr(
            data=df_icc,
            targets="targets",
            raters="rater",
            ratings="rating",
            nan_policy="omit",
        )
        print(icc_results)


if __name__ == "__main__":
    main()
