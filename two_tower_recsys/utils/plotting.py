from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from two_tower_recsys.utils.metrics_history import MetricsHistory


def plot_history(history: MetricsHistory, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.epochs, history.train_loss_epoch, marker="o")
    plt.title("train loss per epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss_epoch.png")
    plt.close()

    plt.figure()
    plt.plot(history.epochs, history.val_recall_10, marker="o")
    plt.title("val recall@10 per epoch")
    plt.xlabel("epoch")
    plt.ylabel("recall@10")
    plt.tight_layout()
    plt.savefig(plots_dir / "val_recall_at_10.png")
    plt.close()

    plt.figure()
    plt.plot(history.epochs, history.val_ndcg_10, marker="o")
    plt.title("val NDCG@10 per epoch")
    plt.xlabel("epoch")
    plt.ylabel("ndcg@10")
    plt.tight_layout()
    plt.savefig(plots_dir / "val_ndcg_at_10.png")
    plt.close()
