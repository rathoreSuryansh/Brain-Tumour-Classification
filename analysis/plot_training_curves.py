"""Loss / accuracy per epoch for every trained model.

    python -m analysis.plot_training_curves

Reads results/<model>/history.json, writes results/figures/loss_accuracy_curves.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR
from common.results import COLORS, load_runs


def main():
    histories = {name: run["history"] for name, run in load_runs().items()}
    colors = COLORS

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, hist in histories.items():
        axes[0].plot(hist['train_loss'], marker='o', label=f'{name} (train)', color=colors[name])
        axes[0].plot(hist['val_loss'], marker='s', linestyle='--', label=f'{name} (val)', color=colors[name], alpha=0.6)

    axes[0].set_title('Loss per Epoch — All Models')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for name, hist in histories.items():
        axes[1].plot(hist['train_acc'], marker='o', label=f'{name} (train)', color=colors[name])
        axes[1].plot(hist['val_acc'], marker='s', linestyle='--', label=f'{name} (val)', color=colors[name], alpha=0.6)

    axes[1].set_title('Accuracy per Epoch — All Models')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / 'loss_accuracy_curves.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
