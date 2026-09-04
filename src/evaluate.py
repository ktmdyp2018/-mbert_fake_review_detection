"""
Evaluation metrics (Section IV.B): Accuracy, Precision, Recall, F1, AUC,
and confusion matrix, matching Tables 6-13 of the manuscript.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

from config import TRAIN
from src.dataset import ReviewDataset, make_collate_fn
from src.train import get_device


@torch.no_grad()
def predict(model, df, batch_size: int = None):
    device = get_device()
    model.eval().to(device)
    collate = make_collate_fn(model.encoder)
    loader = DataLoader(ReviewDataset(df), batch_size=batch_size or TRAIN.batch_size,
                         shuffle=False, collate_fn=collate)

    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device) if batch["token_type_ids"] is not None else None
        handcrafted = batch["handcrafted"].to(device)

        logits, _ = model(input_ids, attention_mask, handcrafted, token_type_ids)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch["labels"].numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": round(100 * accuracy_score(y_true, y_pred), 2),
        "precision": round(100 * precision_score(y_true, y_pred, zero_division=0), 2),
        "recall": round(100 * recall_score(y_true, y_pred, zero_division=0), 2),
        "f1": round(100 * f1_score(y_true, y_pred, zero_division=0), 2),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = round(100 * roc_auc_score(y_true, y_prob), 2)
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def evaluate_model(model, test_df) -> dict:
    y_true, y_pred, y_prob = predict(model, test_df)
    return compute_metrics(y_true, y_pred, y_prob)


def error_analysis(model, test_df) -> dict:
    """Reproduces Table 9-B: characterises false negatives / false
    positives by average valence and emoji density."""
    y_true, y_pred, _ = predict(model, test_df)
    df = test_df.reset_index(drop=True).copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]

    def _avg_valence(sub):
        if len(sub) == 0 or "emotion_vec" not in sub:
            return None
        return float(np.mean([v[0] for v in sub["emotion_vec"]]))

    return {
        "n_false_negatives": int(len(fn)),
        "pct_false_negatives_of_fake": round(100 * len(fn) / max((df["y_true"] == 1).sum(), 1), 2),
        "avg_valence_false_negatives": _avg_valence(fn),
        "n_false_positives": int(len(fp)),
        "pct_false_positives_of_genuine": round(100 * len(fp) / max((df["y_true"] == 0).sum(), 1), 2),
        "avg_valence_false_positives": _avg_valence(fp),
    }
