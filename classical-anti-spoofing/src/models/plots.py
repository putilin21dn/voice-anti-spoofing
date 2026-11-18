from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save_confusion(matrix: list[list[int]], path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks([0, 1], ["human", "machine"])
    axis.set_yticks([0, 1], ["human", "machine"])
    axis.set_xlabel("Предсказание")
    axis.set_ylabel("Метка")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            axis.text(col, row, str(int(values[row, col])), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_roc(curves: list[tuple[str, np.ndarray, np.ndarray]], path) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for name, y_true, scores in curves:
        fpr, tpr, _ = roc_curve(y_true, scores)
        axis.plot(fpr, tpr, label=name)
    axis.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR, recall synthetic")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_pr(curves: list[tuple[str, np.ndarray, np.ndarray]], path) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for name, y_true, scores in curves:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axis.plot(recall, precision, label=name)
    axis.set_xlabel("Recall synthetic")
    axis.set_ylabel("Precision synthetic")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_score_hist(y_true: np.ndarray, scores: np.ndarray, path, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    axis.hist(scores[y_true == 0], bins=20, alpha=0.65, label="human")
    axis.hist(scores[y_true == 1], bins=20, alpha=0.65, label="machine")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Звонки")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
