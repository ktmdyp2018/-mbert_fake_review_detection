"""
Main entry point: reproduces the full experimental pipeline described in
the manuscript (Section IV) end-to-end.

Usage:
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage features
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage stats
    python scripts/run_pipeline.py --stage ablation
    python scripts/run_pipeline.py --stage backbones

Prerequisites:
    1. python scripts/download_datasets.py   (fetches the Hindi set;
       prints instructions for the Kaggle English set)
    2. pip install -r requirements.txt
    3. Network access to huggingface.co, so the real mBERT / XLM-R /
       IndicBERT / MuRIL checkpoints can be downloaded (see README.md).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, TRAIN
from src.dataset import (load_english_dataset, load_hindi_dataset,
                          build_feature_frame, train_val_test_split,
                          cache_feature_frame, load_cached_feature_frame)
from src.train import train_model
from src.evaluate import evaluate_model, error_analysis
from src.statistical_tests import multi_seed_run, mcnemar_test, paired_bootstrap_ci
from src.ablation import run_isolated_ablation, run_cumulative_ablation, run_backbone_substitution
from src.labeling import rule_firing_report


def ensure_dirs():
    for d in (PATHS.cache_dir, PATHS.checkpoint_dir, PATHS.results_dir):
        os.makedirs(d, exist_ok=True)


def stage_features():
    print(">>> Loading + labelling raw datasets ...")
    en_df = load_english_dataset()
    hi_df = load_hindi_dataset()
    print("English rule-firing report:", rule_firing_report(en_df))
    print("Hindi rule-firing report:", rule_firing_report(hi_df))

    print(">>> Extracting handcrafted features (this can take a while on the full corpus) ...")
    en_feat = build_feature_frame(en_df)
    hi_feat = build_feature_frame(hi_df)

    cache_feature_frame(en_feat, "english_full")
    cache_feature_frame(hi_feat, "hindi_full")
    print(f"Cached feature frames to {PATHS.cache_dir}/")
    return en_feat, hi_feat


def _load_or_build_features():
    en_feat = load_cached_feature_frame("english_full")
    hi_feat = load_cached_feature_frame("hindi_full")
    if en_feat is None or hi_feat is None:
        en_feat, hi_feat = stage_features()
    return en_feat, hi_feat


def stage_train_eval():
    en_feat, hi_feat = _load_or_build_features()
    tr_en, va_en, te_en = train_val_test_split(en_feat, seed=42)
    tr_hi, va_hi, te_hi = train_val_test_split(hi_feat, seed=42)

    print(">>> Training on English (in-domain) ...")
    model_en, _ = train_model(tr_en, va_en, seed=42)
    metrics_en = evaluate_model(model_en, te_en)
    print("English in-domain (Table 6):", metrics_en)

    print(">>> Training on Hindi (in-domain) ...")
    model_hi, _ = train_model(tr_hi, va_hi, seed=42)
    metrics_hi = evaluate_model(model_hi, te_hi)
    print("Hindi in-domain (Table 6):", metrics_hi)

    print(">>> Cross-language: English -> Hindi ...")
    metrics_en2hi = evaluate_model(model_en, te_hi)
    print("EN->HI (Table 7):", metrics_en2hi)

    print(">>> Cross-language: Hindi -> English ...")
    metrics_hi2en = evaluate_model(model_hi, te_en)
    print("HI->EN (Table 7):", metrics_hi2en)

    results = {
        "english_in_domain": metrics_en,
        "hindi_in_domain": metrics_hi,
        "en_to_hi": metrics_en2hi,
        "hi_to_en": metrics_hi2en,
        "error_analysis_english": error_analysis(model_en, te_en),
    }
    with open(os.path.join(PATHS.results_dir, "train_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def stage_stats():
    en_feat, hi_feat = _load_or_build_features()
    tr_en, va_en, te_en = train_val_test_split(en_feat, seed=42)
    tr_hi, va_hi, te_hi = train_val_test_split(hi_feat, seed=42)

    print(">>> Multi-seed statistical significance study (Table 6-A) ...")
    stats_en = multi_seed_run(tr_en, va_en, te_en, TRAIN.seeds)
    stats_hi = multi_seed_run(tr_hi, va_hi, te_hi, TRAIN.seeds)
    print("English:", stats_en)
    print("Hindi:", stats_hi)

    with open(os.path.join(PATHS.results_dir, "statistical_significance.json"), "w") as f:
        json.dump({"english": stats_en, "hindi": stats_hi}, f, indent=2, default=str)
    return {"english": stats_en, "hindi": stats_hi}


def stage_ablation():
    en_feat, _ = _load_or_build_features()
    tr_en, va_en, te_en = train_val_test_split(en_feat, seed=42)

    print(">>> Isolated ablation (Table 9-A) ...")
    isolated = run_isolated_ablation(tr_en, va_en, te_en, seed=42)
    print(">>> Cumulative ablation (Tables 8-9) ...")
    cumulative = run_cumulative_ablation(tr_en, va_en, te_en, seed=42)

    with open(os.path.join(PATHS.results_dir, "ablation_results.json"), "w") as f:
        json.dump({"isolated": isolated, "cumulative": cumulative}, f, indent=2)
    return isolated, cumulative


def stage_backbones():
    en_feat, hi_feat = _load_or_build_features()
    tr_en, va_en, _ = train_val_test_split(en_feat, seed=42)
    _, _, te_hi = train_val_test_split(hi_feat, seed=42)

    print(">>> Backbone-substitution study (Table 7-A: EN->HI with mBERT/XLM-R/IndicBERT/MuRIL) ...")
    results = run_backbone_substitution(tr_en, va_en, te_hi, seed=42)
    print(results)

    with open(os.path.join(PATHS.results_dir, "backbone_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "features", "train", "stats",
                                             "ablation", "backbones"], default="all")
    args = parser.parse_args()

    ensure_dirs()

    if args.stage in ("all", "features"):
        stage_features()
    if args.stage in ("all", "train"):
        stage_train_eval()
    if args.stage in ("all", "stats"):
        stage_stats()
    if args.stage in ("all", "ablation"):
        stage_ablation()
    if args.stage in ("all", "backbones"):
        stage_backbones()


if __name__ == "__main__":
    main()
