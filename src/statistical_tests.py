"""
Statistical significance testing (Section IV.C.c, Table 6-A).

  - multi_seed_run(): trains/evaluates the model over TRAIN.seeds and
    reports mean +/- SD and a 95% confidence interval (normal
    approximation over the 5 runs).
  - mcnemar_test(): paired McNemar's test between two models' predictions
    on the same test set.
  - paired_bootstrap_ci(): bootstrap 95% CI for the accuracy DIFFERENCE
    between two models (proposed vs. baseline).
"""

import numpy as np
from scipy import stats

from src.train import train_model
from src.evaluate import evaluate_model, predict


def mean_ci(values, confidence: float = 0.95):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    if len(values) > 1:
        se = sd / np.sqrt(len(values))
        h = se * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    else:
        h = 0.0
    return mean, sd, (mean - h, mean + h)


def multi_seed_run(train_df, val_df, test_df, seeds, backbone: str = None, verbose=True):
    accs, f1s = [], []
    for seed in seeds:
        model, _ = train_model(train_df, val_df, seed=seed, backbone=backbone, verbose=verbose)
        metrics = evaluate_model(model, test_df)
        accs.append(metrics["accuracy"])
        f1s.append(metrics["f1"])
        if verbose:
            print(f"seed={seed}  acc={metrics['accuracy']}  f1={metrics['f1']}")

    acc_mean, acc_sd, acc_ci = mean_ci(accs)
    f1_mean, f1_sd, f1_ci = mean_ci(f1s)
    return {
        "accuracy_mean": round(acc_mean, 2), "accuracy_sd": round(acc_sd, 2), "accuracy_ci": acc_ci,
        "f1_mean": round(f1_mean, 2), "f1_sd": round(f1_sd, 2), "f1_ci": f1_ci,
        "raw_accuracies": accs, "raw_f1s": f1s,
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test comparing model A (e.g. proposed) vs. model B (e.g.
    strongest baseline) on the SAME test set. Returns chi2 statistic and
    p-value using the continuity-corrected formula.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n01 = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n10 = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    if n01 + n10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}

    chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"chi2": round(float(chi2), 3), "p_value": float(p_value), "n01": n01, "n10": n10}


def paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_resamples: int = 10000, seed: int = 0):
    """
    Bootstrap 95% CI for (accuracy_A - accuracy_B) over the test set,
    resampling test examples with replacement.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        acc_a = (y_pred_a[idx] == y_true[idx]).mean()
        acc_b = (y_pred_b[idx] == y_true[idx]).mean()
        diffs[i] = acc_a - acc_b

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff_pp": round(100 * diffs.mean(), 2),
        "ci_95_pp": (round(100 * lower, 2), round(100 * upper, 2)),
        "excludes_zero": bool(lower > 0 or upper < 0),
    }
