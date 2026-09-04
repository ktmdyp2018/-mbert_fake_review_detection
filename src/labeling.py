

import pandas as pd
import numpy as np

from config import LABELING


def _rating_extremity_flag(df: pd.DataFrame) -> pd.Series:
    product_mean = df.groupby("product_id")["rating"].transform("mean")
    is_extreme = df["rating"].isin(LABELING.rating_extreme_values)
    deviates = (df["rating"] - product_mean).abs() >= LABELING.rating_deviation_threshold
    return (is_extreme & deviates).astype(int)


def _burstiness_flag(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("timestamp")
    flags = pd.Series(0, index=df.index)

    for _, group in df.groupby("reviewer_id"):
        ts = group["timestamp"]
        # reviews within a 24h rolling window
        count_24h = ts.apply(
            lambda t: ((ts >= t - pd.Timedelta(hours=LABELING.burstiness_window_hours)) &
                       (ts <= t)).sum()
        )
        # reviews within a 7-day rolling window
        count_7d = ts.apply(
            lambda t: ((ts >= t - pd.Timedelta(days=LABELING.burstiness_window_days)) &
                       (ts <= t)).sum()
        )
        burst = (count_24h >= LABELING.burstiness_min_reviews_24h) | \
                (count_7d >= LABELING.burstiness_min_reviews_7d)
        flags.loc[group.index] = burst.astype(int).values

    return flags.reindex(df.index).fillna(0).astype(int)


def _behavioural_flag(df: pd.DataFrame) -> pd.Series:
    total_reviews_by_reviewer = df.groupby("reviewer_id")["reviewer_id"].transform("count")
    young_account = df["account_age_days"] < LABELING.behavioural_min_account_age_days
    unverified_prolific = (~df["verified_purchase"].astype(bool)) & \
                           (total_reviews_by_reviewer > LABELING.behavioural_min_total_reviews)
    return (young_account | unverified_prolific).astype(int)


def _text_heuristic_flag(df: pd.DataFrame) -> pd.Series:
    """
    Fallback rule used ONLY when reviewer_id/timestamp/account-age
    metadata is unavailable (see note below). Approximates farm-style
    text patterns via repetition and punctuation-exaggeration, computed
    purely from review_text - never from the label-only columns.
    """
    def score(text):
        text = str(text)
        toks = text.split()
        if not toks:
            return 0
        uniq_ratio = len(set(toks)) / len(toks)
        exclaim_density = text.count("!") / max(len(toks), 1)
        very_short = len(toks) <= 3
        return int((uniq_ratio < 0.6 and len(toks) > 4) or exclaim_density > 0.3 or very_short)

    return df["review_text"].apply(score)


def apply_heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schema-adaptive heuristic labelling.

    Full reproduction of the manuscript's 3-rule design (rating
    extremity, burstiness, behavioural pattern; Section III.A) requires
    reviewer_id, timestamp, account_age_days and verified_purchase
    columns. NOTE: the public Hindi release [32] ships only
    {Title, Content, Rating} - it does NOT include reviewer- or
    time-level metadata. When those columns are absent, this function
    automatically falls back to whichever rules ARE computable
    (rating-extremity plus a text-based repetition/exaggeration proxy
    for the missing burstiness/behavioural signals) and prints an
    explicit warning, rather than silently fabricating reviewer
    metadata. For an exact reproduction of Table 2-B, join the raw
    scrape's reviewer-level metadata onto this dataframe before calling
    this function.
    """
    df = df.copy()
    available_rules = {}

    has_rating_cols = {"rating", "product_id"}.issubset(df.columns)
    has_burst_cols = {"reviewer_id", "timestamp"}.issubset(df.columns)
    has_behav_cols = {"reviewer_id", "account_age_days", "verified_purchase"}.issubset(df.columns)

    if has_rating_cols:
        df["rule_rating"] = _rating_extremity_flag(df)
        available_rules["rule_rating"] = df["rule_rating"]
    if has_burst_cols:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["rule_burst"] = _burstiness_flag(df)
        available_rules["rule_burst"] = df["rule_burst"]
    if has_behav_cols:
        df["rule_behav"] = _behavioural_flag(df)
        available_rules["rule_behav"] = df["rule_behav"]

    if not (has_burst_cols and has_behav_cols):
        missing = [c for c, present in
                   [("reviewer_id/timestamp (burstiness)", has_burst_cols),
                    ("reviewer_id/account_age/verified_purchase (behavioural)", has_behav_cols)]
                   if not present]
        print(f"[labeling] WARNING: metadata for {missing} not found in this dataset; "
              f"falling back to a text-based proxy rule for the missing signal(s). "
              f"See src/labeling.py docstring for details.")
        df["rule_text_proxy"] = _text_heuristic_flag(df)
        available_rules["rule_text_proxy"] = df["rule_text_proxy"]

    rule_cols = list(available_rules.keys())
    quorum = max(1, -(-len(rule_cols) // 2))  # ceil(n_rules / 2): majority-of-evidence
    rule_sum = df[rule_cols].sum(axis=1)
    df["label"] = (rule_sum >= quorum).astype(int)  # 1=Fake, 0=Genuine
    df.attrs["rules_used"] = rule_cols
    df.attrs["quorum"] = quorum

    return df


def rule_firing_report(df: pd.DataFrame) -> dict:
    """Diagnostic report reproducing the rule-firing proportions discussed
    in the revised manuscript (Section III.A)."""
    fake = df[df["label"] == 1]
    n = max(len(fake), 1)
    return {
        "n_fake": int(len(fake)),
        "n_genuine": int((df["label"] == 0).sum()),
        "pct_rating_extremity": round(100 * fake["rule_rating"].sum() / n, 1),
        "pct_burstiness": round(100 * fake["rule_burst"].sum() / n, 1),
        "pct_behavioural": round(100 * fake["rule_behav"].sum() / n, 1),
    }


def validate_against_human_annotations(df: pd.DataFrame, human_col: str = "human_label"):
    """
    Cohen's kappa agreement between heuristic labels and a manually
    verified subsample (Section III.A validation study). `human_col`
    should contain NaN for unverified rows and 0/1 for the verified
    stratified subsample.
    """
    from sklearn.metrics import cohen_kappa_score, accuracy_score

    verified = df.dropna(subset=[human_col])
    if verified.empty:
        return {"n_verified": 0, "kappa": None, "agreement": None}

    kappa = cohen_kappa_score(verified[human_col].astype(int), verified["label"].astype(int))
    agreement = accuracy_score(verified[human_col].astype(int), verified["label"].astype(int))
    return {
        "n_verified": int(len(verified)),
        "kappa": round(float(kappa), 3),
        "agreement": round(float(agreement) * 100, 1),
    }
