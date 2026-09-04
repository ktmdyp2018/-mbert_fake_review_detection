

from typing import List
import copy
import torch

from config import FEATURE_DIMS
from src.train import train_model
from src.evaluate import evaluate_model

# Index ranges within the 11-dim handcrafted vector [emotion(7) ++ linguistic(4)]
EMOTION_SLICE = slice(0, 5)          # Val, Aro, EI, EP, EIC
EMOJI_SLICE = slice(5, 7)            # EmojiEM_val, EmojiEM_aro
LINGUISTIC_SLICE = slice(7, 10)      # length, repetition, readability  (excludes ASV)
ASV_SLICE = slice(10, 11)            # ASV (ABSA-derived)

MODULE_SLICES = {
    "emotion": EMOTION_SLICE,
    "emoji": EMOJI_SLICE,
    "absa": ASV_SLICE,
    "linguistic": LINGUISTIC_SLICE,
}


class _MaskedForwardWrapper:
    """Context-manager style monkeypatch that zeroes selected handcrafted-
    feature slices and/or the F_cross block for the duration of a run,
    without modifying FullPipelineModel's source."""

    def __init__(self, model, active_modules: List[str], use_cross_dataset: bool = True):
        self.model = model
        self.active_modules = set(active_modules)
        self.use_cross_dataset = use_cross_dataset
        self._orig_forward = model.forward

    def _masked_forward(self, input_ids, attention_mask, handcrafted, token_type_ids=None,
                         paired_input_ids=None, paired_attention_mask=None):
        handcrafted = handcrafted.clone()
        for name, sl in MODULE_SLICES.items():
            if name not in self.active_modules:
                handcrafted[:, sl] = 0.0

        C_i, _ = self.model.encoder(input_ids, attention_mask, token_type_ids)
        if self.use_cross_dataset and "cross_dataset" in self.active_modules:
            F_cross = self.model.cross_module.interpolate(C_i, None)
        else:
            F_cross = torch.zeros_like(C_i)

        fused = torch.cat([C_i, handcrafted, F_cross], dim=-1)
        logits = self.model.classifier(fused)
        return logits, C_i

    def __enter__(self):
        self.model.forward = self._masked_forward
        return self.model

    def __exit__(self, *exc):
        self.model.forward = self._orig_forward


ALL_MODULES = ["emotion", "emoji", "absa", "linguistic", "cross_dataset"]


def run_isolated_ablation(train_df, val_df, test_df, seed: int = 42) -> dict:
    """Table 9-A: mBERT-only baseline, then each module added ALONE."""
    results = {}

    # mBERT-only baseline: train with no handcrafted modules active.
    model, _ = train_model(train_df, val_df, seed=seed, verbose=False)
    with _MaskedForwardWrapper(model, active_modules=[]):
        results["mbert_only"] = evaluate_model(model, test_df)

    for module in ALL_MODULES:
        model, _ = train_model(train_df, val_df, seed=seed, verbose=False)
        with _MaskedForwardWrapper(model, active_modules=[module]):
            results[f"+{module}_only"] = evaluate_model(model, test_df)

    model, _ = train_model(train_df, val_df, seed=seed, verbose=False)
    with _MaskedForwardWrapper(model, active_modules=ALL_MODULES):
        results["full_fusion"] = evaluate_model(model, test_df)

    return results


def run_cumulative_ablation(train_df, val_df, test_df, seed: int = 42,
                             order: List[str] = None) -> dict:
    """Tables 8-9: modules added sequentially in `order`."""
    order = order or ["emotion", "emoji", "absa", "linguistic", "cross_dataset"]
    results = {}
    active = []

    model, _ = train_model(train_df, val_df, seed=seed, verbose=False)
    with _MaskedForwardWrapper(model, active_modules=active):
        results["mbert_only"] = evaluate_model(model, test_df)

    for module in order:
        active = active + [module]
        model, _ = train_model(train_df, val_df, seed=seed, verbose=False)
        with _MaskedForwardWrapper(model, active_modules=active):
            results["+".join(active)] = evaluate_model(model, test_df)

    return results


def run_backbone_substitution(train_en, val_en, test_hi, seed: int = 42) -> dict:
    """Table 7-A: swap the mBERT backbone for XLM-R / IndicBERT / MuRIL,
    training on English and testing on Hindi (English->Hindi transfer)."""
    from config import BACKBONE_CHECKPOINTS
    results = {}
    for backbone in BACKBONE_CHECKPOINTS:
        model, _ = train_model(train_en, val_en, seed=seed, backbone=backbone, verbose=False)
        results[backbone] = evaluate_model(model, test_hi)
    return results
