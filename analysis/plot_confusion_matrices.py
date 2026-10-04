"""2x2 grid of test-set confusion matrices, one per trained model.

    python -m analysis.plot_confusion_matrices

Reads results/<model>/metrics.json, writes results/figures/confusion_matrices.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common.config import CLASSES, FIGURES_DIR
from common.results import load_runs


def main():
    runs = load_runs()
    cms = {name: np.array(run["metrics"]["confusion_matrix"]) for name, run in runs.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (name, cm) in zip(axes.flat, cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f'{name} — Confusion Matrix')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for ax in list(axes.flat)[len(cms):]:   # hide unused panels while some models are still missing
        ax.axis('off')

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / 'confusion_matrices.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
