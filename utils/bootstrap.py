#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap test for ICC difference between two LLM-as-a-judge models vs clinicians.

This script:
1) Loads clinician ratings and aggregates to clinician_mean per sample (Answer key)
2) Loads Claude judge outputs and parses 3 rubric scores -> claude_mean per sample
3) Loads Meditron judge outputs and parses "Score" -> meditron_mean per sample
4) Merges on Answer (must match across files)
5) Computes ICC(3) for (clinician_mean, claude_mean) and (clinician_mean, meditron_mean)
6) Bootstraps ΔICC = ICC_claude - ICC_meditron with percentile 95% CI

Usage example:
python bootstrap_icc_diff.py \
  --clinicians_jsonl /path/clinicians.jsonl \
  --clinician_answer_key Answer \
  --clinician_score_key clinician_score \
  --claude_jsonl /path/claude_results_v3.jsonl \
  --claude_answer_key orig_response \
  --claude_response_key response \
  --meditron_jsonl /path/output_with_generated-v10.jsonl \
  --meditron_answer_key orig_response \
  --meditron_generation_key raw_generation \
  --n_boot 5000

Dependencies:
pip install pandas numpy pingouin scipy
"""

import argparse
import json
import re
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pingouin as pg


# -----------------------------
# Parsing helpers
# -----------------------------
def extract_claude_three_scores(response_text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract numeric scores for:
      - Alignment with guidelines
      - Relevance and/or Completeness
      - Harmlessness
    Returns (alignment, relevance, harmlessness) as ints, or None if not found.
    """
    patterns = {
        "alignment": r"Alignment with guidelines[\s\S]*?(?:[Ss]core[^\d]*)?(\d+)",
        "relevance": r"Relevance\s*(?:and|&)\s*Completeness[\s\S]*?(?:[Ss]core[^\d]*)?(\d+)",
        "harmlessness": r"Harmlessness[\s\S]*?(?:[Ss]core[^\d]*)?(\d+)",
    }
    flags = re.IGNORECASE | re.DOTALL

    def find_score(pattern: str) -> Optional[int]:
        m = re.search(pattern, response_text, flags)
        return int(m.group(1)) if m else None

    a = find_score(patterns["alignment"])
    r_ = find_score(patterns["relevance"])
    h = find_score(patterns["harmlessness"])
    return a, r_, h


def parse_first_score_from_generation(generated_text: str) -> Optional[float]:
    """
    Finds the first numeric value after `"Score":` in generated_text.
    Accepts ints or floats, returns float or None.
    """
    match_scores = re.findall(r'"Score"\s*:\s*([-+]?\d*\.\d+|\d+)', generated_text)
    if not match_scores:
        return None
    try:
        return float(match_scores[0])
    except Exception:
        return None


# -----------------------------
# ICC + bootstrap
# -----------------------------
def icc3_between(df: pd.DataFrame, col1: str, col2: str, target_col: str = "Answer") -> float:
    """
    Compute ICC(3) between two raters (col1 and col2) over targets (target_col).
    """
    long_df = pd.melt(
        df[[target_col, col1, col2]],
        id_vars=[target_col],
        value_vars=[col1, col2],
        var_name="rater",
        value_name="rating",
    )
    #.dropna(subset=["rating"])

    #if long_df[target_col].nunique() < 2:
     #   return float("nan")

    icc_tbl = pg.intraclass_corr(
        data=long_df,
        targets=target_col,
        raters="rater",
        ratings="rating",
        nan_policy="omit",
    )
    return float(icc_tbl.loc[icc_tbl["Type"] == "ICC3k", "ICC"].values[0])


def bootstrap_icc_diff(
    df: pd.DataFrame,
    modelA_col: str,
    modelB_col: str,
    human_col: str = "clinician_mean",
    target_col: str = "Answer",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap ΔICC = ICC(human, modelA) - ICC(human, modelB) by resampling targets (rows) with replacement.
    Returns observed ICCs, diff, CI, and significance.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    print(df)
    df.to_csv("bootstrap_med.csv")
    print(modelA_col)
    icc_A = icc3_between(df, human_col, modelA_col, target_col)
    print(modelB_col)
    icc_B = icc3_between(df, human_col, modelB_col, target_col)
    obs_diff = icc_A - icc_B

    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample targets
        boot = df.iloc[idx]
        a = icc3_between(boot, human_col, modelA_col, target_col)
        c = icc3_between(boot, human_col, modelB_col, target_col)
        diffs[b] = a - c

    ci_low, ci_high = np.nanpercentile(diffs, [2.5, 97.5])
    significant = not (ci_low <= 0 <= ci_high)

    return {
        "icc_modelA": icc_A,
        "icc_modelB": icc_B,
        "diff_A_minus_B": obs_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(significant),
        # Uncomment if you want to save distribution later:
        # "boot_diffs": diffs,
    }
def extract_json_text(s: str) -> str:
    # 1) Prefer fenced code block
    m = re.search(r"```(?:[\w-]+)?\s*(.*?)```", s, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    # 2) Fallback: first '{' to last '}'
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()

# -----------------------------
# Loaders
# -----------------------------
def load_clinicians_jsonl(path: str, answer_key: str, score_key: str) -> pd.DataFrame:
    """
    Load clinician ratings JSONL. Expects per-rater rows with:
      - answer_key (e.g., "Answer" or "orig_response")
      - score_key  (e.g., "clinician_score" or "score")
    Returns one row per Answer with clinician_mean.
    """
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            if answer_key not in data or score_key not in data:
                continue
            ans = data[answer_key]
            sc = data[score_key]
            try:
                sc = float(sc)
            except Exception:
                continue
            rows.append({"Answer": ans, "clinician_score": sc})

    if not rows:
        raise ValueError(f"No clinician rows loaded from {path}. Check keys: {answer_key}, {score_key}")

    df = pd.DataFrame(rows)
    df_agg = (
        df.groupby("Answer", as_index=False)["clinician_score"]
        .mean()
        .rename(columns={"clinician_score": "clinician_mean"})
    )
    return df_agg
def load_clinicians_csv(path: str, answer_key: str, score_key: str) -> pd.DataFrame:
    """
    Load clinician ratings from CSV.
    Expects columns:
      - answer_key (e.g., "Answer")
      - score_key  (e.g., "clinician_score")
    Returns one row per Answer with clinician_mean.
    """
    df = pd.read_csv(path)

    #if answer_key not in df.columns:
     #   raise ValueError(f"{answer_key} not found in clinician CSV columns: {df.columns}")

    #if score_key not in df.columns:
     #   raise ValueError(f"{score_key} not found in clinician CSV columns: {df.columns}")
    score_cols = [
    'Score_Alignment_with_guidelines',
    'Score_Relevance_and_completeness',
    'Score Harmlessness'
    #'First Alignment with guidelines',
    #'First Relevance & completeness',
    #'First Harmlessness',
    ]

    # Mean across the 3 criteria
    df["clinician_mean"] = df[score_cols].mean(axis=1)
    
    df_agg = df[["Answer", "clinician_mean"]].dropna(subset=["clinician_mean"])
    #df_agg = (
       # df.groupby(answer_key, as_index=False)[score_key]
      #  .mean()
     #   .rename(columns={answer_key: "Answer", score_key: "clinician_mean"})
    #)

    return df_agg

def load_claude_jsonl(path: str, answer_key: str, response_key: str) -> pd.DataFrame:
    """
    Load Claude judge JSONL. Expects:
      - answer_key: key to join on (should match Meditron + clinicians), often "orig_response"
      - response_key: verbose judge text to parse, often "response"
    Returns one row per Answer with claude_mean.
    """
    rows = []
    claude_scores = []
    n_fail = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            resp = data[response_key]
            a, r_, h = extract_claude_three_scores(resp)
            mean = None
            if None not in (a, r_, h):
                mean = (a + r_ + h) / 3.0
            else:
                n_fail += 1

            claude_scores.append(mean)

        df_scores = pd.read_json("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/dataset_ref_based_scores_test.jsonl", lines=True)
        for i, row in df_scores.iterrows():
            rows.append({"Answer": row["orig_response"], "claude_mean": claude_scores[i]})

    df = pd.DataFrame(rows)
    print(df)
    print(f"[Claude] Loaded {len(df)} rows. Parsing failures (missing any of 3 scores or keys): {n_fail}")
    return df
def load_gpt_jsonl(path: str, answer_key: str, response_key: str) -> pd.DataFrame:

    rows = []
    gpt_scores = []
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
        # --- usage ---
            raw_output = data["response"]["body"]["choices"][0]["message"]["content"]  # your long string
            try:
                raw_output_clean = extract_json_text(raw_output)
                parsed_output = json.loads(raw_output_clean)
                scores = {k: v.get("score") for k, v in parsed_output.items() if isinstance(v, dict) and "score" in v}
                alignment = scores.get("Alignment with guidelines")
                relevance = scores.get("Relevance and completeness")
                harmlessness = scores.get("Harmlessness")
            #print("Extracted scores - Alignment:", alignment, "Relevance:", relevance, "Harmlessness:", harmlessness)
            except Exception as e:
                print("Error extracting scores:", e)
                alignment, relevance, harmlessness = None, None, None


            if None not in (alignment, relevance, harmlessness):
                mean_score = (alignment + relevance + harmlessness) / 3
            else:
                mean_score = None
        #gpt_scores.append(harmlessness)
            gpt_scores.append(mean_score)

    df_scores = pd.read_json("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/dataset_ref_based_scores_test.jsonl", lines=True)
    for i, row in df_scores.iterrows():
        rows.append({"Answer": row["orig_response"], "gpt_mean": gpt_scores[i]})

    df = pd.DataFrame(rows)
    print(df)
    #print(f"[GPT] Loaded {len(df)} rows. Parsing failures (missing any of 3 scores or keys): {n_fail}")
    return df


def load_meditron_jsonl(path: str, answer_key: str, generation_key: str) -> pd.DataFrame:
    """
    Load Meditron judge JSONL. Expects:
      - answer_key: key to join on, often "orig_response"
      - generation_key: generation text containing `"Score": <number>`, often "raw_generation"
    Returns one row per Answer with meditron_mean (float).
    """
    rows = []
    n_fail = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            if answer_key not in data or generation_key not in data:
                n_fail += 1
                continue

            ans = data[answer_key]
            gen = data[generation_key]

            score = parse_first_score_from_generation(gen)
            if score is None:
                n_fail += 1

            rows.append({"Answer": ans, "meditron_mean": score})

    df = pd.DataFrame(rows)
    print(f"[Meditron] Loaded {len(df)} rows. Parsing failures (no Score or missing keys): {n_fail}")
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinicians_jsonl", required=True)
    ap.add_argument("--clinician_answer_key", default="Answer")
    ap.add_argument("--clinician_score_key", default="clinician_score")

    ap.add_argument("--claude_jsonl", required=True)
    ap.add_argument("--claude_answer_key", default="orig_response")
    ap.add_argument("--claude_response_key", default="response")

    ap.add_argument("--meditron_jsonl", required=True)
    ap.add_argument("--meditron_answer_key", default="orig_response")
    ap.add_argument("--meditron_generation_key", default="raw_generation")

    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default=None)

    args = ap.parse_args()

    # Load
    df_clin = load_clinicians_csv(args.clinicians_jsonl, args.clinician_answer_key, args.clinician_score_key)
    print(df_clin)
    #df_claude = load_claude_jsonl(args.claude_jsonl, args.claude_answer_key, args.claude_response_key)
    df_claude = load_gpt_jsonl(args.claude_jsonl, args.claude_answer_key, args.claude_response_key)
    df_medi = load_meditron_jsonl(args.meditron_jsonl, args.meditron_answer_key, args.meditron_generation_key)
    print(df_medi)
    print([repr(c) for c in df_medi.columns])
    df_medi_agg = (
        df_medi
        .groupby("Answer", as_index=False)
        .agg(meditron_mean=("meditron_mean", "mean"))
    )
    df_claude_agg = (
        df_claude
        .groupby("Answer", as_index=False)
        .agg(claude_mean=("gpt_mean", "mean"))
    )
    
    # Merge (one row per Answer)
    df = (
        df_clin
        .merge(df_claude_agg, on="Answer", how="inner")
        .merge(df_medi_agg, on="Answer", how="inner")
        .dropna(subset=["clinician_mean", "claude_mean", "meditron_mean"])
    )

    print(f"[Merge] Rows after merge+dropna: {len(df)}")

    # Sanity check duplicates
    dup = df["Answer"].duplicated().sum()
    if dup:
        print(f"[Warn] {dup} duplicated Answer keys after merge. Consider using a stable sample_id instead of text.")

    # ICC + bootstrap ΔICC
    res = bootstrap_icc_diff(
        df,
        modelA_col="claude_mean",
        modelB_col="meditron_mean",
        human_col="clinician_mean",
        target_col="Answer",
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print("\n=== Results ===")
    print(f"ICC(3) clinician vs Claude   : {res['icc_modelA']:.4f}")
    print(f"ICC(3) clinician vs Meditron : {res['icc_modelB']:.4f}")
    print(f"ΔICC (Claude - Meditron)     : {res['diff_A_minus_B']:.4f}")
    print(f"95% CI (percentile bootstrap): [{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
    print(f"Significant (CI excludes 0)  : {res['significant']}")

    # Optional output merged table for debugging
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved merged table to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
