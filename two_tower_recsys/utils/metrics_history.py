from __future__ import annotations

from dataclasses import dataclass, field

from lightning import Callback, Trainer
from lightning.pytorch.core.module import LightningModule


@dataclass
class MetricsHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss_epoch: list[float] = field(default_factory=list)
    val_recall_10: list[float] = field(default_factory=list)
    val_ndcg_10: list[float] = field(default_factory=list)


class MetricsHistoryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.history = MetricsHistory()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = int(trainer.current_epoch)
        metrics = trainer.callback_metrics

        self.history.epochs.append(epoch)

        if "train/loss_epoch" in metrics:
            self.history.train_loss_epoch.append(float(metrics["train/loss_epoch"].cpu().item()))
        else:
            self.history.train_loss_epoch.append(float("nan"))

        if "val/recall@10" in metrics:
            self.history.val_recall_10.append(float(metrics["val/recall@10"].cpu().item()))
        else:
            self.history.val_recall_10.append(float("nan"))

        if "val/ndcg@10" in metrics:
            self.history.val_ndcg_10.append(float(metrics["val/ndcg@10"].cpu().item()))
        else:
            self.history.val_ndcg_10.append(float("nan"))
