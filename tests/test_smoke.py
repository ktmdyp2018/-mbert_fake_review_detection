

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN, PATHS
from src.dataset import (load_hindi_dataset, build_feature_frame,
                          train_val_test_split, cache_feature_frame)
from src.train import train_model
from src.evaluate import evaluate_model, error_analysis
from src.ablation import run_isolated_ablation
from src.statistical_tests import mcnemar_test, paired_bootstrap_ci


def make_synthetic_english(n=60, seed=0):
    """
    Synthetic English sample used ONLY because the real Kaggle dataset
    [31] requires manual authentication and cannot be fetched in this
    environment. Labels are assigned directly from the generative
    `is_fake` flag (rather than recomputed via the heuristic rules) so
    the smoke test has a guaranteed, non-degenerate class balance to
    exercise the training loop; the heuristic-labelling logic itself is
    already validated separately against the REAL Hindi dataset in
    STEP 1. For real experiments, use `src.dataset.load_english_dataset`
    on the actual Kaggle CSV, which DOES run the full heuristic pipeline.
    """
    rng = np.random.default_rng(seed)
    good = ["Great product, works perfectly and arrived fast.",
            "I love this item, excellent quality and durable.",
            "Very satisfied, exactly as described, would recommend."]
    bad = ["Amazing amazing amazing buy now best product ever!!!",
           "Best product best product best product 5 stars!!!",
           "Terrible quality broke after one day, waste of money."]
    rows = []
    for i in range(n):
        is_fake = rng.random() < 0.4
        text = rng.choice(bad if is_fake else good)
        rating = int(rng.choice([1, 5])) if is_fake else int(rng.choice([3, 4, 5]))
        rows.append({
            "product_id": f"P{i % 5}",
            "reviewer_id": f"R{i}",
            "review_text": text,
            "rating": rating,
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            "account_age_days": int(rng.integers(1, 500)),
            "verified_purchase": bool(rng.random() > 0.3),
            "label": int(is_fake),
            "lang": "en",
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("STEP 1: Load real Hindi dataset + build synthetic English sample")
    print("=" * 70)
    os.makedirs(PATHS.data_dir, exist_ok=True)

    if os.path.exists(PATHS.hindi_xlsx):
        hi_full = load_hindi_dataset()
        hi_small = hi_full.sample(min(80, len(hi_full)), random_state=0).reset_index(drop=True)
        print(f"Loaded real Hindi dataset, using {len(hi_small)}-row sample. "
              f"Rules used: {hi_full.attrs.get('rules_used')}")
    else:
        raise SystemExit("Hindi dataset not found - run scripts/download_datasets.py first.")

    from src.labeling import apply_heuristic_labels
    en_small = make_synthetic_english(n=60)
    print(f"Built synthetic English sample: {len(en_small)} rows "
          f"(label balance: {en_small['label'].mean():.2f} fake)")

    print("\n" + "=" * 70)
    print("STEP 2: Feature extraction (emotion + emoji + ABSA + linguistic)")
    print("=" * 70)
    hi_feat = build_feature_frame(hi_small)
    en_feat = build_feature_frame(en_small)
    cache_feature_frame(hi_feat, "smoke_hindi")
    cache_feature_frame(en_feat, "smoke_english")
    print("Feature frames built OK. Example emotion_vec:", hi_feat.iloc[0]["emotion_vec"])

    print("\n" + "=" * 70)
    print("STEP 3: Train/val/test split")
    print("=" * 70)
    tr_en, va_en, te_en = train_val_test_split(en_feat, seed=42)
    tr_hi, va_hi, te_hi = train_val_test_split(hi_feat, seed=42)
    print(f"English split: {len(tr_en)}/{len(va_en)}/{len(te_en)}")
    print(f"Hindi split:   {len(tr_hi)}/{len(va_hi)}/{len(te_hi)}")

    print("\n" + "=" * 70)
    print("STEP 4: Train the full model for a couple of epochs (English)")
    print("=" * 70)
    TRAIN.epochs = 2  # keep the smoke test fast
    TRAIN.batch_size = 8
    model, history = train_model(tr_en, va_en, seed=42, verbose=True)
    print("Training loop completed OK. History:", history)
    print("Encoder is_mock:", model.encoder.is_mock)

    print("\n" + "=" * 70)
    print("STEP 5: Evaluate on English test set + Hindi (cross-language)")
    print("=" * 70)
    metrics_en = evaluate_model(model, te_en)
    print("English in-domain metrics:", metrics_en)
    metrics_cross = evaluate_model(model, te_hi)
    print("English -> Hindi cross-language metrics:", metrics_cross)

    print("\n" + "=" * 70)
    print("STEP 6: Error analysis (Table 9-B style)")
    print("=" * 70)
    print(error_analysis(model, te_en))

    print("\n" + "=" * 70)
    print("STEP 7: Statistical significance utilities (McNemar / bootstrap)")
    print("=" * 70)
    from src.evaluate import predict
    y_true, y_pred_a, _ = predict(model, te_en)
    # Compare the model against a "random" pseudo-baseline for a functional check
    y_pred_b = np.random.default_rng(0).integers(0, 2, size=len(y_true))
    print("McNemar:", mcnemar_test(y_true, y_pred_a, y_pred_b))
    print("Bootstrap CI:", paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_resamples=500))

    print("\n" + "=" * 70)
    print("STEP 8: Isolated ablation (Table 9-A style, reduced scope)")
    print("=" * 70)
    ablation_results = run_isolated_ablation(tr_en, va_en, te_en, seed=42)
    for k, v in ablation_results.items():
        print(f"  {k}: acc={v['accuracy']}  f1={v['f1']}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED: full pipeline runs end-to-end without errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()
