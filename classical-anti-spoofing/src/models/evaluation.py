from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true_machine: np.ndarray, y_score_machine: np.ndarray, y_pred_machine: np.ndarray) -> dict:
    """Положительный класс — machine/synthetic. Score не является калиброванной вероятностью."""

    y_true = np.asarray(y_true_machine).astype(int)
    y_pred = np.asarray(y_pred_machine).astype(int)
    y_score = np.asarray(y_score_machine, dtype=float)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    result = {
        "human": int((y_true == 0).sum()),
        "machine": int((y_true == 1).sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "recall_synthetic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_human": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_synthetic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_synthetic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "fnr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "confusion_matrix_labels": ["human", "machine"],
        "confusion_matrix": matrix.tolist(),
        "score_note": "score — ранжирующая оценка, не калиброванная вероятность",
    }
    if len(np.unique(y_true)) == 2 and np.isfinite(y_score).all() and len(np.unique(y_score)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result
