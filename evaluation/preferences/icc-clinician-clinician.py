import itertools
import numpy as np
import pandas as pd
import pingouin as pg


# -----------------------------
# Helpers
# -----------------------------
def icc3k_two_raters(df_wide, target_col, r1_col, r2_col, min_items=2):
    """
    Compute ICC(3,k) (Pingouin label: ICC3k) for two 'raters' from a wide df with:
      [target_col, r1_col, r2_col]
    Returns float or None.
    """
    df_wide = df_wide.dropna(subset=[r1_col, r2_col])
    if len(df_wide) < min_items:
        return None

    df_long = df_wide.melt(
        id_vars=target_col,
        value_vars=[r1_col, r2_col],
        var_name="rater",
        value_name="rating",
    )

    if df_long[target_col].nunique() < min_items:
        return None

    try:
        icc_tbl = pg.intraclass_corr(
            data=df_long,
            targets=target_col,
            raters="rater",
            ratings="rating",
        )
        val = icc_tbl.loc[icc_tbl["Type"] == "ICC3k", "ICC"].values[0]
        if np.isfinite(val) and -1 <= val <= 1:
            return float(val)
    except Exception:
        return None

    return None


def bootstrap_mean_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap CI for the mean of a list/array (ignores NaNs)."""
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
# Load + build preference table
# -----------------------------
df = pd.read_csv("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/CHUV_2025-09-29_pseudononym_KT_v2.csv")

# Encode preference as 1 = A (First Answer), 0 = B (Second Answer)
df["pref01"] = df["Vote"].apply(lambda x: 1 if x == 1 else 0).astype(float)

df["item_id"] = (
    df["First Answer"].astype(str).str.strip()
    + " ||| "
    + df["Second Answer"].astype(str).str.strip()
)

df["rater"] = df["Name_coded"].astype(str)

# One row per (item_id, rater)
pref_df = (
    df[["item_id", "rater", "pref01"]]
    .dropna(subset=["item_id", "rater", "pref01"])
    .groupby(["item_id", "rater"], as_index=False)["pref01"]
    .mean()
)
pref_df["pref01"] = (pref_df["pref01"] >= 0.5).astype(float)

raters = pref_df["rater"].unique()


# -----------------------------
# 1) Pairwise clinician–clinician ICC(3,k)
# -----------------------------
pair_iccs = []
pairs = list(itertools.combinations(raters, 2))

for r1, r2 in pairs:
    d1 = pref_df.loc[pref_df["rater"] == r1, ["item_id", "pref01"]].rename(columns={"pref01": "r1"})
    d2 = pref_df.loc[pref_df["rater"] == r2, ["item_id", "pref01"]].rename(columns={"pref01": "r2"})
    m = pd.merge(d1, d2, on="item_id", how="inner")  # only shared items

    icc_val = icc3k_two_raters(m, target_col="item_id", r1_col="r1", r2_col="r2", min_items=2)
    if icc_val is not None:
        pair_iccs.append(icc_val)

pair_mean, pair_ci = bootstrap_mean_ci(pair_iccs, n_boot=1000, seed=1)


# -----------------------------
# 2) Leave-one-out clinician vs consensus ICC(3,k)
#    consensus = mean of OTHER clinicians' 0/1 preferences per item (excluding the clinician)
# -----------------------------
loo_iccs = []

for r in raters:
    r_df = pref_df.loc[pref_df["rater"] == r, ["item_id", "pref01"]].rename(columns={"pref01": "rater_pref"})

    others = pref_df.loc[pref_df["rater"] != r, ["item_id", "pref01"]]

    # mean of others -> continuous in [0,1] (this is good for an ICC-like comparison)
    consensus = (
        others.groupby("item_id", as_index=False)["pref01"]
        .mean()
        .rename(columns={"pref01": "consensus_pref"})
    )

    # only items rated by r AND at least one other clinician
    m = pd.merge(r_df, consensus, on="item_id", how="inner")

    icc_val = icc3k_two_raters(
        m,
        target_col="item_id",
        r1_col="rater_pref",
        r2_col="consensus_pref",
        min_items=2,
    )
    if icc_val is not None:
        loo_iccs.append(icc_val)

loo_mean, loo_ci = bootstrap_mean_ci(loo_iccs, n_boot=1000, seed=2)


# -----------------------------
# Print
# -----------------------------
print("=== Preferences encoded as 0/1; ICC computed on numeric ratings ===")

print("\n--- Pairwise clinician–clinician ICC(3,k) ---")
print(f"pairs total: {len(pairs)}")
print(f"pairs used:  {len(pair_iccs)}")
print(f"mean ICC3k:  {pair_mean:.3f}")
print(f"95% CI:      [{pair_ci[0]:.3f}, {pair_ci[1]:.3f}]")

print("\n--- Leave-one-out clinician vs consensus ICC(3,k) ---")
print(f"clinicians total: {len(raters)}")
print(f"clinicians used:  {len(loo_iccs)}")
print(f"mean ICC3k:       {loo_mean:.3f}")
print(f"95% CI:           [{loo_ci[0]:.3f}, {loo_ci[1]:.3f}]")

