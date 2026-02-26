import itertools
import numpy as np
import pandas as pd
import pingouin as pg

# -----------------------------
# Helpers
# -----------------------------
def compute_icc3k_from_two_columns(df_wide, target_col="answer", r1_col="r1", r2_col="r2"):
    """
    Compute ICC(3,k) (Pingouin: ICC3k) for two raters from a wide df:
    columns: [target_col, r1_col, r2_col]
    Returns float or None if cannot compute.
    """
    df_wide = df_wide.dropna(subset=[r1_col, r2_col])
    if len(df_wide) < 2:
        return None

    df_long = df_wide.melt(
        id_vars=target_col,
        value_vars=[r1_col, r2_col],
        var_name="rater",
        value_name="rating",
    )

    # With 2 raters, len(df_long) = 2 * n_items
    if len(df_long) < 5:
        return None

    try:
        icc_tbl = pg.intraclass_corr(
            data=df_long,
            targets=target_col,
            raters="rater",
            ratings="rating",
        )
        icc_val = icc_tbl.loc[icc_tbl["Type"] == "ICC3k", "ICC"].values[0]
        if np.isfinite(icc_val) and -1 <= icc_val <= 1:
            return float(icc_val)
    except Exception:
        return None

    return None


def bootstrap_mean_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap CI for the mean of a list/array."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), (float(lo), float(hi))


# -----------------------------
# Load + preprocess
# -----------------------------
df = pd.read_csv("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/CHUV_2025-07-15_anonymised.csv")

score_cols = [
    "First Answer Alignment with Guidelines",
    "First Answer Relevance & Completeness",
    "First Answer Harmlessness",
]
# uncomment the following if you want to Z-normalize within each clinician user_id
df[score_cols] = df.groupby("User ID")[score_cols].transform(
    lambda x:x
    #(x - x.mean()) / x.std(ddof=0)
)

# Choose the voted answer
df["answer"] = np.where(df["Vote"] == 1, df["First Answer"], df["Second Answer"])

# Composite score
df["mean_score"] = df[score_cols].mean(axis=1)

# Build clinician table: one row per (answer, rater)
clin_df = (
    df[["answer", "mean_score", "User ID"]]
    .rename(columns={"User ID": "rater"})
    .astype({"rater": str})
    .groupby(["answer", "rater"], as_index=False)["mean_score"]
    .mean()
    .dropna(subset=["mean_score"])
)

clinicians = clin_df["rater"].unique()

# -----------------------------
# 1) Pairwise clinician–clinician ICC (avg over pairs)
# -----------------------------
pair_iccs = []
pairs = list(itertools.combinations(clinicians, 2))

for r1, r2 in pairs:
    sub1 = clin_df.loc[clin_df["rater"] == r1, ["answer", "mean_score"]].rename(columns={"mean_score": "r1"})
    sub2 = clin_df.loc[clin_df["rater"] == r2, ["answer", "mean_score"]].rename(columns={"mean_score": "r2"})

    merged = pd.merge(sub1, sub2, on="answer", how="inner")  # only shared answers
    icc_val = compute_icc3k_from_two_columns(merged, target_col="answer", r1_col="r1", r2_col="r2")
    if icc_val is not None:
        pair_iccs.append(icc_val)

pair_mean, pair_ci = bootstrap_mean_ci(pair_iccs, n_boot=1000, seed=1)

# -----------------------------
# 2) Clinician vs leave-one-out consensus ICC (avg over clinicians)
# -----------------------------
consensus_iccs = []

# Pre-split per rater for speed
by_rater = {r: clin_df[clin_df["rater"] == r][["answer", "mean_score"]] for r in clinicians}

for r in clinicians:
    # Clinician r's ratings
    r_df = by_rater[r].rename(columns={"mean_score": "r1"})

    # Consensus excluding r:
    # For each answer, mean of other raters' mean_score (must have >=1 other rater)
    others = clin_df[clin_df["rater"] != r]
    consensus = (
        others.groupby("answer", as_index=False)["mean_score"]
        .mean()
        .rename(columns={"mean_score": "r2"})
    )

    # Keep only answers rated by r AND at least one other clinician (so r2 exists)
    merged = pd.merge(r_df, consensus, on="answer", how="inner")

    icc_val = compute_icc3k_from_two_columns(merged, target_col="answer", r1_col="r1", r2_col="r2")
    if icc_val is not None:
        consensus_iccs.append(icc_val)

cons_mean, cons_ci = bootstrap_mean_ci(consensus_iccs, n_boot=1000, seed=2)

# -----------------------------
# Print results
# -----------------------------
print("=== Clinician–Clinician Pairwise ICC (avg over pairs) ===")
print(f"n_pairs_used: {len(pair_iccs)} / {len(pairs)}")
print(f"Mean ICC3k: {pair_mean:.3f}")
print(f"95% bootstrap CI: [{pair_ci[0]:.3f}, {pair_ci[1]:.3f}]")

print("\n=== Clinician vs Leave-One-Out Consensus ICC (avg over clinicians) ===")
print(f"n_clinicians_used: {len(consensus_iccs)} / {len(clinicians)}")
print(f"Mean ICC3k: {cons_mean:.3f}")
print(f"95% bootstrap CI: [{cons_ci[0]:.3f}, {cons_ci[1]:.3f}]")

