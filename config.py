"""
Central configuration for the Multilingual mBERT + Emotion-Aware Fake Review
Detection Framework.

All values here mirror the specification tables added to the revised
manuscript:
    Table 2-A / 2-B  -> dataset & label-feature independence config
    Table 5-A        -> FNN classifier architecture
    Table 5-B        -> training hyperparameters
    Table 7-A        -> backbone-substitution options
"""

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
@dataclass
class PathConfig:
    data_dir: str = "data"
    english_csv: str = "data/english_reviews.csv"        # from Kaggle [31]
    hindi_xlsx: str = "data/Hindi_Fake_Review_Dataset.xlsx"  # from GitHub [32]
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


# --------------------------------------------------------------------------
# Backbone encoder options (Table 7-A)
# --------------------------------------------------------------------------
BACKBONE_CHECKPOINTS = {
    "mbert": "bert-base-multilingual-cased",   # 104 languages, 119,547 WordPiece vocab
    "xlmr": "xlm-roberta-base",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "muril": "google/muril-base-cased",
}


# --------------------------------------------------------------------------
# Heuristic labelling rules (Section III.A, Table 2-B)
# --------------------------------------------------------------------------
@dataclass
class LabelingConfig:
    rating_extreme_values: tuple = (1, 5)
    rating_deviation_threshold: float = 2.0        # |rating - product_mean| >= 2
    burstiness_window_hours: int = 24
    burstiness_min_reviews_24h: int = 5
    burstiness_window_days: int = 7
    burstiness_min_reviews_7d: int = 15
    behavioural_min_account_age_days: int = 30
    behavioural_min_total_reviews: int = 10
    fake_rule_quorum: int = 2                      # >= 2 of 3 rules => "Fake"
    # Signals below are used ONLY for label construction and must NEVER be
    # passed to the model as input features (label/feature independence).
    label_only_columns: List[str] = field(
        default_factory=lambda: ["rating", "timestamp", "account_age_days",
                                  "verified_purchase", "reviewer_id"]
    )


# --------------------------------------------------------------------------
# Feature dimensionalities (Section III.G / Table 5-A)
# --------------------------------------------------------------------------
@dataclass
class FeatureDims:
    mbert_cls: int = 768
    emotion: int = 7          # Val, Aro, EI, EP, EIC, EmojiEM_val, EmojiEM_aro
    linguistic: int = 4       # length, repetition_ratio, readability, ASV
    cross_dataset: int = 768  # F_cross (domain-adapted contextual vector)

    @property
    def fused_dim(self) -> int:
        return self.mbert_cls + self.emotion + self.linguistic + self.cross_dataset  # 1547


# --------------------------------------------------------------------------
# FNN classifier architecture (Table 5-A)
# --------------------------------------------------------------------------
@dataclass
class FNNConfig:
    fc1_out: int = 768
    fc2_out: int = 256
    fc3_out: int = 64
    num_classes: int = 2
    dropout_fc1: float = 0.3
    dropout_fc2: float = 0.3
    dropout_fc3: float = 0.2


# --------------------------------------------------------------------------
# Training hyperparameters (Table 5-B)
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    backbone: str = "mbert"
    max_seq_len: int = 256
    batch_size: int = 16
    epochs: int = 8
    early_stop_patience: int = 3

    mbert_lr: float = 2e-5
    mbert_weight_decay: float = 0.01
    fnn_lr: float = 5e-4

    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0

    # Only the last N transformer layers are fine-tuned; the rest are frozen.
    num_frozen_layers: int = 8       # layers 1-8 frozen, 9-12 fine-tuned (of 12)

    # Domain-adversarial schedule (Eq. 9b) - Ganin & Lempitsky (2015)
    grl_gamma: float = 10.0

    # Auxiliary interpolation regulariser (Eq. 9a)
    cross_dataset_alpha: float = 0.5

    seeds: List[int] = field(default_factory=lambda: [13, 42, 101, 2023, 777])

    device: str = "cuda"  # falls back to "cpu" automatically if unavailable


# --------------------------------------------------------------------------
# ABSA configuration (Section III.E)
# --------------------------------------------------------------------------
@dataclass
class ABSAConfig:
    aspects: List[str] = field(
        default_factory=lambda: ["price", "quality", "delivery",
                                  "packaging", "service", "durability"]
    )
    sentence_transformer_ckpt: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.6


# --------------------------------------------------------------------------
# Train/val/test split ratios (used consistently for the proposed model
# AND for every re-implemented baseline, to guarantee an apple-to-apple
# comparison; see Section IV.E).
# --------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

PATHS = PathConfig()
LABELING = LabelingConfig()
FEATURE_DIMS = FeatureDims()
FNN = FNNConfig()
TRAIN = TrainConfig()
ABSA = ABSAConfig()
