"""
Dataset loading and feature-frame construction (Section III.A/III.B).

`load_english_dataset` / `load_hindi_dataset` read the raw CSV/XLSX files
(Kaggle [31] / GitHub [32]), apply heuristic labelling (src/labeling.py),
and hand back a DataFrame with a `label` column plus the *raw* text -
handcrafted feature extraction happens afterwards in `build_feature_frame`,
enforcing label/feature independence (Table 2-B): `label_only_columns`
are dropped before feature extraction ever sees the frame.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from config import PATHS, LABELING, SPLIT_RATIOS
from src.preprocessing import preprocess_review
from src.labeling import apply_heuristic_labels
from src.emotion_features import compute_emotion_vector
from src.absa_features import AspectMatcher, AspectSentimentScorer, compute_absa
from src.linguistic_features import compute_linguistic_vector


# --------------------------------------------------------------------------
# Raw loaders
# --------------------------------------------------------------------------
def _auto_map_columns(df: pd.DataFrame, mapping_candidates: dict) -> pd.DataFrame:
    """
    Renames whichever candidate column name is present to the canonical
    name expected downstream (review_text, rating, product_id, ...).
    `mapping_candidates`: {canonical_name: [list, of, candidate, names]}
    """
    df = df.copy()
    for canonical, candidates in mapping_candidates.items():
        if canonical in df.columns:
            continue
        for cand in candidates:
            if cand in df.columns:
                df = df.rename(columns={cand: canonical})
                break
    return df


ENGLISH_COLUMN_CANDIDATES = {
    "review_text": ["review_text", "Review Text", "reviewText", "text", "review", "Review"],
    "rating": ["rating", "Rating", "star_rating", "overall"],
    "product_id": ["product_id", "asin", "ProductId", "product"],
    "reviewer_id": ["reviewer_id", "reviewerID", "user_id", "reviewer"],
    "timestamp": ["timestamp", "reviewTime", "date", "Date", "review_date"],
    "account_age_days": ["account_age_days", "account_age"],
    "verified_purchase": ["verified_purchase", "verified"],
}

HINDI_COLUMN_CANDIDATES = {
    "review_text": ["review_text", "Content (Hindi)", "Content", "content", "Review", "text"],
    "title": ["Title (Hindi)", "Title", "title"],
    "rating": ["rating", "Rating"],
    "product_id": ["product_id", "ProductId", "product"],
    "reviewer_id": ["reviewer_id", "reviewerID", "user_id"],
    "timestamp": ["timestamp", "date", "Date"],
    "account_age_days": ["account_age_days", "account_age"],
    "verified_purchase": ["verified_purchase", "verified"],
    "sentiment_hint": ["Label", "label", "Sentiment"],
}


def load_english_dataset(path: str = None) -> pd.DataFrame:
    """
    Loads the Kaggle English review dataset [31]. Column names are
    auto-mapped from several common variants (see
    ENGLISH_COLUMN_CANDIDATES) since different Kaggle mirrors of this
    dataset use slightly different headers. If `product_id` is absent it
    is synthesised as a constant so the rating-extremity rule still runs
    (comparing each rating to the *global* mean rather than the
    per-product mean in that degraded case - a warning is printed).
    """
    path = path or PATHS.english_csv
    df = pd.read_csv(path)
    df = _auto_map_columns(df, ENGLISH_COLUMN_CANDIDATES)

    if "review_text" not in df.columns:
        raise ValueError(
            f"Could not find a review-text column in {path}. "
            f"Expected one of {ENGLISH_COLUMN_CANDIDATES['review_text']}."
        )
    if "product_id" not in df.columns:
        print("[dataset] WARNING: no product_id column found; rating-extremity "
              "rule will use the global mean rating instead of per-product mean.")
        df["product_id"] = "GLOBAL"

    df["lang"] = "en"
    return apply_heuristic_labels(df)


def load_hindi_dataset(path: str = None) -> pd.DataFrame:
    """
    Loads the public Hindi Fake-Review dataset [32]. NOTE: the public
    release only ships {#, Title (Hindi), Content (Hindi), Rating,
    Label}; there is no reviewer_id/timestamp/account_age metadata, and
    the `Label` column is a 3-class SENTIMENT annotation (Positive/
    Neutral/Negative), not a Fake/Genuine label - it is kept only as an
    informational `sentiment_hint` column and is never used as the
    target label or as a model feature. See src/labeling.py for how the
    heuristic label is derived under this reduced metadata.
    """
    path = path or PATHS.hindi_xlsx
    df = pd.read_excel(path)
    df = _auto_map_columns(df, HINDI_COLUMN_CANDIDATES)

    if "review_text" not in df.columns:
        raise ValueError(
            f"Could not find a review-text column in {path}. "
            f"Expected one of {HINDI_COLUMN_CANDIDATES['review_text']}."
        )
    if "title" in df.columns:
        df["review_text"] = (df["title"].fillna("") + ". " + df["review_text"].fillna("")).str.strip(". ")
    if "product_id" not in df.columns:
        df["product_id"] = "GLOBAL"

    df["lang"] = "hi"
    return apply_heuristic_labels(df)


def train_val_test_split(df: pd.DataFrame, seed: int = 42):
    """Stratified 70/15/15 split (SPLIT_RATIOS), reused identically for
    the proposed model and every re-implemented baseline (Section IV.E)."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=(1 - SPLIT_RATIOS["train"]), stratify=df["label"], random_state=seed
    )
    val_ratio_of_temp = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp), stratify=temp_df["label"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# --------------------------------------------------------------------------
# Handcrafted feature-frame construction (emotion + linguistic/ABSA)
# --------------------------------------------------------------------------
def build_feature_frame(df: pd.DataFrame, use_attention_aggregation: bool = False) -> pd.DataFrame:
    """
    Enforces label/feature independence: `label_only_columns` (rating,
    timestamp, account_age_days, verified_purchase, reviewer_id) are
    dropped from the working frame BEFORE any feature is computed, so
    they can never leak into the model, even accidentally.

    Adds columns: clean_text, emotion_vec (7,), linguistic_vec (4,).
    """
    work = df.drop(columns=[c for c in LABELING.label_only_columns if c in df.columns],
                    errors="ignore").copy()

    matcher = AspectMatcher()
    scorer = AspectSentimentScorer()

    clean_texts, emotion_vecs, linguistic_vecs = [], [], []

    for _, row in tqdm(work.iterrows(), total=len(work), desc="Extracting features"):
        lang = row.get("lang", "en")
        pre = preprocess_review(row["review_text"], lang=lang)

        absa = compute_absa(pre["sentences"], lang, matcher, scorer)
        emo_vec = compute_emotion_vector(pre["sentences"], pre["emojis"], lang=lang)
        ling_vec = compute_linguistic_vector(pre["tokens"], pre["clean_text"],
                                              asv=absa["asv"], lang=lang)

        clean_texts.append(pre["clean_text"])
        emotion_vecs.append(emo_vec)
        linguistic_vecs.append(ling_vec)

    work["clean_text"] = clean_texts
    work["emotion_vec"] = emotion_vecs
    work["linguistic_vec"] = linguistic_vecs
    return work


def cache_feature_frame(df: pd.DataFrame, name: str):
    os.makedirs(PATHS.cache_dir, exist_ok=True)
    out_path = os.path.join(PATHS.cache_dir, f"{name}.pkl")
    df.to_pickle(out_path)
    return out_path


def load_cached_feature_frame(name: str):
    path = os.path.join(PATHS.cache_dir, f"{name}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None


# --------------------------------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------------------------------
class ReviewDataset(Dataset):
    """
    Wraps a feature-frame (output of `build_feature_frame`). `__getitem__`
    returns raw text (tokenised lazily by the collate_fn using the active
    ContextualEncoder's tokenizer) plus the pre-computed 11-dim
    handcrafted-feature vector [emotion(7) ++ linguistic(4)] and the
    integer label.
    """

    def __init__(self, feature_df: pd.DataFrame):
        self.df = feature_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        handcrafted = np.concatenate([row["emotion_vec"], row["linguistic_vec"]]).astype(np.float32)
        domain_label = 0 if row.get("lang", "en") == "en" else 1
        return {
            "text": row["clean_text"],
            "handcrafted": handcrafted,
            "label": int(row["label"]),
            "domain_label": domain_label,
        }


def make_collate_fn(encoder):
    import torch

    def collate(batch):
        texts = [b["text"] for b in batch]
        enc = encoder.tokenizer(
            texts, padding=True, truncation=True,
            max_length=encoder.max_seq_len, return_tensors="pt",
        )
        handcrafted = torch.tensor(np.stack([b["handcrafted"] for b in batch]), dtype=torch.float32)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        domain_labels = torch.tensor([b["domain_label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids"),
            "handcrafted": handcrafted,
            "labels": labels,
            "domain_labels": domain_labels,
        }

    return collate


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Weighted cross-entropy weights for the class-imbalanced datasets
    (Section IV.A: ~65/35 genuine/fake split on English). Always returns
    a length-2 array (index 0 = Genuine, 1 = Fake) even if one class is
    entirely absent from `labels` (e.g. a very small sample split), to
    keep nn.CrossEntropyLoss's `weight` argument shape-consistent with
    the model's 2-class output head.
    """
    weights = np.ones(2, dtype=np.float32)
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * cnt)
    return weights
