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

    df["_target"] = (
        df["response_A"].astype(str) + " || " + df["response_B"].astype(str)
    )

    long_df = pd.melt(
        df,
        id_vars=["_target"],
        value_vars=[col1, col2],
        var_name="rater",
        value_name="rating",
    )
    #.dropna(subset=["rating"])

    long_df["rating"] = long_df["rating"].map({'A': 1, 'B': 0})
    #if long_df[target_col].nunique() < 2:
     #   return float("nan")

    icc_tbl = pg.intraclass_corr(
        data=long_df,
        targets="_target",
        raters="rater",
        ratings="rating",
        nan_policy="omit",
    )
    print("je suis la")
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

def mean_result(results: list) -> float:
    """
    Converts list of 'A'/'B' to mean score: A=1, B=0.
    """
    count_a = sum(1 for r in results if r == 'A')
    count_b = sum(1 for r in results if r == 'B')
    total = count_a + count_b
    if count_a > count_b:
        return "A"
    elif count_b > count_a:
        return "B"
    else:
        return "Tie"
def load_clin_pref (path:str):
    df = pd.read_csv(path)
    df["clinician_mean"] = df["First Answer Improved"].apply(lambda x: 'A' if pd.notna(x) else 'B')

    df_clinicians = df.groupby(["First Answer", "Second Answer"], as_index=False)["clinician_mean"].agg(mean_result)
    df_clinicians = df_clinicians.rename(columns={
    "First Answer": "response_A",
    "Second Answer": "response_B"
})
    return df_clinicians

def load_gpt_pref_jsonl (path:str):
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
                scores = parsed_output["Answer"]
            
            except Exception as e:
                print("Error extracting scores:", e)
                scores = None
                marche_pas += 1
            gpt_scores.append(scores)

    df_pref = pd.read_json("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/dataset_ref_based_pref_test_new_v2.jsonl", lines=True)
    for i, row in df_pref.iterrows():
        rows.append({"response_A": row["orig_response_A"],
                    "response_B": row["orig_response_B"], "claude_mean": gpt_scores[i]})
    df = pd.DataFrame(rows)

    return df


def load_pref_jsonl(path:str):
    
    rows = []
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            response_A = data["orig_response_A"]
            response_B= data["orig_response_B"]
            result = data["winner"]

            rows.append({
                "response_A": response_A,
                "response_B": response_B,
                "meditron_mean": result,
            })
        df = pd.DataFrame(rows)
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
    #df_clin = load_clinicians_csv(args.clinicians_jsonl, args.clinician_answer_key, args.clinician_score_key)
    #print(df_clin)
    #df_claude = load_claude_jsonl(args.claude_jsonl, args.claude_answer_key, args.claude_response_key)
    #df_claude = load_gpt_jsonl(args.claude_jsonl, args.claude_answer_key, args.claude_response_key)
    #df_medi = load_meditron_jsonl(args.meditron_jsonl, args.meditron_answer_key, args.meditron_generation_key)
    #print(df_medi)

    df_clin = load_clin_pref(args.clinicians_jsonl)
    df_claude = load_gpt_pref_jsonl(args.claude_jsonl)
    df_medi = load_pref_jsonl(args.meditron_jsonl)
    print([repr(c) for c in df_medi.columns])

    
    # Merge (one row per Answer)
    df = (
        df_clin
        .merge(df_claude, on=["response_A", "response_B"], how="inner")
        .merge(df_medi, on=["response_A", "response_B"], how="inner")
        .dropna(subset=["clinician_mean", "claude_mean", "meditron_mean"])
    )

    print(f"[Merge] Rows after merge+dropna: {len(df)}")


    # ICC + bootstrap ΔICC
    res = bootstrap_icc_diff(
        df,
        modelA_col="claude_mean",
        modelB_col="meditron_mean",
        human_col="clinician_mean",
        target_col=["response_A", "response_B"],
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
