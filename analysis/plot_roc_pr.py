"""Micro-averaged ROC and precision-recall curves for every trained model.

    python -m analysis.plot_roc_pr

Needs results/<model>/test_outputs.npz (written by each model notebook's last
cell). Writes results/figures/roc_comparison.png and pr_comparison.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from common.config import CLASSES, FIGURES_DIR
from common.results import COLORS, load_runs


def main():
    runs = load_runs(need_outputs=True)
    colors = COLORS

    test_labels = next(iter(runs.values()))["labels"]
    for name, run in runs.items():
        assert np.array_equal(run["labels"], test_labels), f"{name}: test labels differ between models"

    y_true_bin = label_binarize(test_labels, classes=list(range(len(CLASSES))))
    model_probs = {name: run["probs"] for name, run in runs.items()}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- ROC ---------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=colors[name], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Micro-Averaged) — Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / 'roc_comparison.png', dpi=150)
    plt.close()

    # ---- Precision-Recall --------------------------------------------------
    plt.figure(figsize=(9, 7))
    for name, probs in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), probs.ravel())
        ap = average_precision_score(y_true_bin, probs, average='micro')
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', color=colors[name], linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Micro-Averaged) — Model Comparison')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / 'pr_comparison.png', dpi=150)
    plt.close()
    print(f"Saved ROC and PR curves to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
