from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from two_tower_recsys.data.datamodule import MovieLensPairsDataModule
from two_tower_recsys.data.download import ensure_raw_zip_exists
from two_tower_recsys.data.preprocess import preprocess_movielens
from two_tower_recsys.models.lightning_module import TwoTowerLightningModule
from two_tower_recsys.utils.git import get_git_commit_id
from two_tower_recsys.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


class SingleBatchDataModule(MovieLensPairsDataModule):
    def val_dataloader(self):
        dummy = torch.zeros((1,), dtype=torch.long)
        ds = torch.utils.data.TensorDataset(dummy)
        return torch.utils.data.DataLoader(ds, batch_size=1)


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    raw_zip = ensure_raw_zip_exists(cfg.paths.raw_zip)

    preprocess_movielens(
        raw_zip_path=str(raw_zip),
        processed_dir=cfg.paths.processed_dir,
        rating_threshold=float(cfg.preprocess.rating_threshold),
        min_user_interactions=int(cfg.preprocess.min_user_interactions),
        min_item_interactions=int(cfg.preprocess.min_item_interactions),
        max_interactions=int(cfg.preprocess.max_interactions),
        val_last_n=int(cfg.preprocess.val_last_n),
        test_last_n=int(cfg.preprocess.test_last_n),
    )

    train_path = str(Path(cfg.paths.processed_dir) / "train.parquet")
    val_path = str(Path(cfg.paths.processed_dir) / "val.parquet")

    datamodule = SingleBatchDataModule(
        train_path=train_path,
        val_path=val_path,
        batch_size=int(cfg.model.data.batch_size),
        num_workers=int(cfg.model.data.num_workers),
    )
    datamodule.setup()

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.logging.mlflow.tracking_uri),
        experiment_name=str(cfg.logging.mlflow.experiment_name),
        run_name=str(cfg.run.name),
    )
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", get_git_commit_id())

    val_df = pd.read_parquet(val_path)

    module = TwoTowerLightningModule(
        num_users=datamodule.artifacts.num_users,
        num_items=datamodule.artifacts.num_items,
        embedding_dim=int(cfg.model.embedding_dim),
        dropout=float(cfg.model.dropout),
        temperature=float(cfg.model.temperature),
        lr=float(cfg.model.optimizer.lr),
        weight_decay=float(cfg.model.optimizer.weight_decay),
        k_list=list(cfg.metrics.k_list),
        filter_seen=bool(cfg.metrics.filter_seen),
        seen_items_by_user=datamodule.artifacts.seen_items_by_user,
        val_df=val_df,
    )

    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        deterministic=bool(cfg.trainer.deterministic),
        logger=mlflow_logger,
        enable_checkpointing=bool(cfg.trainer.enable_checkpointing),
    )

    trainer.fit(module, datamodule=datamodule)

    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # берём последнее значение train/loss_epoch если есть
    metrics = trainer.callback_metrics
    if "train/loss_epoch" in metrics:
        loss_val = float(metrics["train/loss_epoch"].cpu().item())
        plt.figure()
        plt.title("Final train loss (epoch)")
        plt.bar(["loss"], [loss_val])
        plt.tight_layout()
        plt.savefig(plots_dir / "final_train_loss.png")
        plt.close()

    artifacts_dir = Path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    item_emb = module.model.item_embedding.weight.detach().cpu()
    torch.save(item_emb, artifacts_dir / "item_embeddings.pt")


if __name__ == "__main__":
    main()
