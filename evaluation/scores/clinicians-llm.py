"""
===============================================================================
LLM vs Clinician Evaluation Script (Unified Version)
===============================================================================

This script evaluates agreement between clinician scores and LLM-generated
scores. It supports multiple LLM output formats via either built-in logic
(fine-tuning mode) or external parser modules.

-------------------------------------------------------------------------------
SUPPORTED MODELS
-------------------------------------------------------------------------------

--model must be one of:

    fine-tuning  → Built-in score extraction using --file_type or --score_field
    meditron     → Uses external parser: parse_meditron.py
    prometheus   → Uses external parser: parse_prometheus.py
    claude       → Uses external parser: parse_claude.py
    gpt          → Uses external parser: parse_gpt.py
    llama        → Uses external parser: parse_llama.py

For all models except "fine-tuning", a file named:

    parse_<model>.py

must exist in the same directory and define:

    parse_<model>(record) -> (answer, score) oe (score)

-------------------------------------------------------------------------------
LLM SCORE EXTRACTION (fine-tuning mode only)
-------------------------------------------------------------------------------

If --model fine-tuning is selected, score extraction is controlled by:

1) --file_type feedback-1-score
      → Extracts score from data["raw_generation"]
        using regex pattern: "Score": <number>

2) --file_type feedback-3-score
      → Extracts scores from data["raw_generation"] using regex patterns for each score.

3) --score_field <field_name>
      → Directly extracts score from data[<field_name>]

NOTE:
    If --score_field is provided, it overrides --file_type.

The answer is taken from:
    data["orig_response"]
or, if missing, extracted from a tag inside data["instruction"].

-------------------------------------------------------------------------------
INPUT FILES
-------------------------------------------------------------------------------

--llm_jsonl
    Path to JSONL file containing LLM outputs.

--clinician_csv (optional argument but internally currently hardcoded path)
    CSV file containing clinician evaluations with columns:

        "Answer"
        "Name"
        "Score_Alignment_with_guidelines"
        "Score_Relevance_and_completeness"
        "Score Harmlessness"

Clinician mean score is computed as the mean of the three score columns.

-------------------------------------------------------------------------------
SUPPORTED METRICS
-------------------------------------------------------------------------------

--metric mse
    Mean Squared Error computed over merged_df.
    Includes bootstrap 95% confidence interval.

--metric kappa
    Quadratic-weighted Cohen's kappa computed on merged_clin.
    Scores are rounded to integers before calculation.

--metric icc
    Intraclass Correlation Coefficient (ICC3)
    computed using pingouin.intraclass_corr on merged_clin.

-------------------------------------------------------------------------------
EXAMPLES
-------------------------------------------------------------------------------

Fine-tuning with feedback extraction and MSE:

    python new_pers-clinicians-llm.py \
        --llm_jsonl outputs.jsonl \
        --model fine-tuning \
        --file_type feedback-1-score \
        --metric mse

Meditron with ICC:

    python new_pers-clinicians-llm.py \
        --llm_jsonl meditron_outputs.jsonl \
        --model meditron \
        --metric icc

Prometheus with weighted kappa:

    python new_pers-clinicians-llm.py \
        --llm_jsonl prometheus_outputs.jsonl \
        --model prometheus \
        --metric kappa

-------------------------------------------------------------------------------
NOTES
-------------------------------------------------------------------------------

• Matching between LLM and clinicians is done via the "Answer" column.
• All LLM parsers must return (answer, score).
• ICC requires at least 2 shared answers per clinician.
• Kappa uses quadratic weighting.

===============================================================================
"""


import re
import json
import argparse
import importlib
from xml.parsers.expat import model
import pandas as pd
import pingouin as pg
import numpy as np
from sklearn.metrics import cohen_kappa_score
from utils.extract_scores import extract_scores

def compute_mean_score(data: dict, file_type: str=None, score_field: str = None):
    if file_type == "feedback-3-score":
        # call extract scores from utils folder
        return np.mean(extract_scores(data.get("raw_generation", ""))) if extract_scores(data.get("raw_generation", "")) is not None else None
        

    if file_type == "feedback-1-score":
        generated_text = data.get("raw_generation", "")
        match_scores = re.findall(r'"Score"\s*:\s*([-+]?\d*\.\d+|\d+)', generated_text)
        if not match_scores:
            scores = data.get("extracted_scores", [])
            clean_scores=[]
            if scores:
                clean_scores = [s for s in scores if isinstance(s, (int, float))]

            if len(clean_scores) == 0:
                return None

            return float(np.mean(clean_scores))
        try:
            return int(float(match_scores[0]))
        except Exception:
            return None

    if score_field:
        return np.mean(data.get(score_field, None)) if data.get(score_field, None) is not None else None

    raise ValueError(f"Unknown file_type: {file_type}")

def load_external_parser(model: str):
    """
    Loads parse_<model>.py and returns parse_<model> function.

    Example for model 'meditron':
      module: parse_meditron
      function: parse_meditron(record) -> (answer, score)
    """
    module_name = f"parse_{model}"
    func_name = f"parse_{model}"
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise ImportError(
            f"Module '{module_name}.py' does not define '{func_name}(record)'."
        )
    return fn



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_jsonl", type=str, required=True)
    parser.add_argument(
        "--file_type",
        type=str,
        required=False,
        choices=["feedback-1-score", "feedback-3-score"],
    )
    parser.add_argument("--score_field", type=str, required=False, help="Field name for direct score extraction (if file_type is not feedback or self-consistency)")
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["mse", "kappa", "icc"],
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fine-tuning", "meditron", "prometheus", "jury", "claude", "gpt", "llama"],
        help="Model name for LLM output parsing. Use 'fine-tuning' for built-in parsing logic based on --file_type and --score_field. For other models, an external parser module named parse_<model>.py must be present.",
    )
    parser.add_argument("--clinician_csv", type=str, required=True, help="Path to clinician CSV")

    args = parser.parse_args()

    # --------------------------
    # LLM outputs -> df_llm_agg
    # --------------------------
    scores = []
    parsing_failures = 0
    df = pd.read_csv(args.clinician_csv)

    if args.model == "fine-tuning":
        parse_fn = None  
    else:
        parse_fn = load_external_parser(args.model)
    with open(args.llm_jsonl, "r") as file:
        for (i,row), line in zip(df.iterrows(),file):
            data = json.loads(line)
            if args.model == "fine-tuning":
                mean_score = compute_mean_score(data, args.file_type, args.score_field)
                orig_response = data.get("orig_response", None)
                if orig_response is None:
                    match = re.search(r"<\s*response\s*>(.*?)<\s*/\s*response\s*>", data["instruction"], flags=re.DOTALL | re.IGNORECASE)
                    orig_response = match.group(1).strip() if match else None
                if orig_response is None:
                    match = re.search(r'###Response:\s*(.*?)\s*###Reference Answer', data["instruction"], flags=re.DOTALL)
                    orig_response = match.group(1).strip() if match else None
            
            else:
                try:
                    if args.model == "meditron" or args.model == "prometheus":
                        orig_response, mean_score = parse_fn(data)
                    else:
                        orig_response= row["Answer"]
                        mean_score= parse_fn(data)

                except Exception as e:
                    print("Error parsing record:", e)
                    print("Record content:", data)
                    orig_response, mean_score = None, None

                
            scores.append({"Answer": orig_response, "llm_mean": mean_score})
        

    df_llm = pd.DataFrame(scores)
    print(f"Number of parsing failures: {parsing_failures} (missing orig_response)")

    df_llm_agg = (
        df_llm.dropna(subset=["llm_mean"])
        .groupby("Answer", as_index=False)["llm_mean"]
        .mean()
    )

    # --------------------------
    # Clinician CSV -> df + df_clinicians
    # --------------------------
    df = pd.read_csv(args.clinician_csv)
    score_cols = [
        "Score_Alignment_with_guidelines",
        "Score_Relevance_and_completeness",
        "Score Harmlessness",
    ]
    df["clinician_mean"] = df[score_cols].mean(axis=1)

    # per-clinician table (needed for ICC/Kappa)
    df_clinicians = df[["Answer", "clinician_mean"]].dropna(subset=["clinician_mean"])

    # --------------------------
    # Build TWO merges:
    #   - merged_df for MSE (full df)
    #   - merged_clin for ICC/Kappa (df_clinicians)
    # --------------------------
    merged_df = pd.merge(df, df_llm_agg, on="Answer", how="inner").dropna(
        subset=["clinician_mean", "llm_mean"]
    )
    merged_clin = pd.merge(df_clinicians, df_llm_agg, on="Answer", how="inner").dropna(
        subset=["clinician_mean", "llm_mean"]
    )

    print("len merged_df (full df):", len(merged_df))
    print("len merged_clin (per clinician):", len(merged_clin))

    # --------------------------
    # Metric computations
    # --------------------------
    if args.metric == "mse":
        # MSE computed over FULL df merge (as you requested)
        sq_error = (merged_df["clinician_mean"] - merged_df["llm_mean"]) ** 2
        mse = float(sq_error.mean())
        print(f"MSE over merged_df rows: {mse:.6f}")

        # Bootstrap CI over rows (optional)
        rng = np.random.default_rng(42)
        n_boot = 10000
        arr = sq_error.to_numpy()
        n = len(arr)
        if n == 0:
            print("No rows to compute MSE.")
            return
        boot = []
        for _ in range(n_boot):
            sample = arr[rng.integers(0, n, size=n)]
            boot.append(sample.mean())
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    elif args.metric == "kappa":
        # Kappa computed on PER-CLINICIAN merge
        clinician_int = merged_clin["clinician_mean"].round().astype(int)
        llm_int = merged_clin["llm_mean"].round().astype(int)

        kappa = cohen_kappa_score(clinician_int, llm_int, weights="quadratic")
        print(f"Cohen's kappa (rounded ints) on merged_clin: {kappa:.6f}")

    elif args.metric == "icc":
        # ICC computed on PER-CLINICIAN merge
        df_icc = pd.melt(
            merged_clin,
            id_vars=["Answer"],
            value_vars=["clinician_mean", "llm_mean"],
            var_name="rater",
            value_name="rating",
        )

        icc_results = pg.intraclass_corr(
            data=df_icc,
            targets="Answer",
            raters="rater",
            ratings="rating",
            nan_policy="omit",
        )
        print(icc_results)



if __name__ == "__main__":
    main()

