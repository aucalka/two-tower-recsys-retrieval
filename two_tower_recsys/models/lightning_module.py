from __future__ import annotations

import pandas as pd
import torch
from lightning import LightningModule
from torch import nn

from two_tower_recsys.eval.metrics import ndcg_at_k, recall_at_k
from two_tower_recsys.models.two_tower import TwoTower


class TwoTowerLightningModule(LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        dropout: float,
        temperature: float,
        lr: float,
        weight_decay: float,
        k_list: list[int],
        filter_seen: bool,
        seen_items_by_user: dict[int, set[int]],
        val_df: pd.DataFrame | None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["seen_items_by_user", "val_df"])

        self.model = TwoTower(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.k_list = list(k_list)
        self.filter_seen = bool(filter_seen)

        self.seen_items_by_user = seen_items_by_user
        self.val_df = val_df

        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        user_idx, pos_item_idx = batch
        user_emb = self.model.encode_users(user_idx)
        item_emb = self.model.encode_items(pos_item_idx)

        logits = (user_emb @ item_emb.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.loss_fn(logits, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def validation_step(self, batch: object, batch_idx: int) -> None:
        if batch_idx != 0:
            return
        if self.val_df is None:
            return

        val_df = self.val_df
        user_idx = torch.tensor(val_df["user_idx"].to_numpy(), dtype=torch.long, device=self.device)
        true_item_idx = torch.tensor(
            val_df["item_idx"].to_numpy(), dtype=torch.long, device=self.device
        )

        user_emb = self.model.encode_users(user_idx)
        all_item_emb = self.model.item_embedding.weight.to(self.device)

        scores = user_emb @ all_item_emb.T

        if self.filter_seen:
            for row, u in enumerate(user_idx.tolist()):
                seen = self.seen_items_by_user.get(int(u))
                if seen:
                    scores[row, list(seen)] = -1e9

        true_scores = scores.gather(1, true_item_idx.view(-1, 1))
        ranks = (scores > true_scores).sum(dim=1)

        for k in self.k_list:
            self.log(f"val/recall@{k}", recall_at_k(ranks, k), prog_bar=True, on_epoch=True)
        self.log("val/ndcg@10", ndcg_at_k(ranks, 10), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
