from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from two_tower_recsys.data.datasets import PairDataset


@dataclass(frozen=True)
class DataModuleArtifacts:
    num_users: int
    num_items: int
    seen_items_by_user: dict[int, set[int]]


class MovieLensPairsDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_ds: PairDataset | None = None
        self._val_df: pd.DataFrame | None = None
        self.artifacts: DataModuleArtifacts | None = None

    def setup(self, stage: str | None = None) -> None:
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)

        num_users = int(train_df["user_idx"].max()) + 1
        num_items = int(train_df["item_idx"].max()) + 1

        user_tensor = torch.tensor(train_df["user_idx"].to_numpy(), dtype=torch.long)
        item_tensor = torch.tensor(train_df["item_idx"].to_numpy(), dtype=torch.long)
        self._train_ds = PairDataset(user_tensor, item_tensor)

        self._val_df = val_df

        seen_items_by_user: dict[int, set[int]] = {}
        for user_idx, item_idx in zip(
            train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy()
        ):
            user_idx_int = int(user_idx)
            item_idx_int = int(item_idx)
            if user_idx_int not in seen_items_by_user:
                seen_items_by_user[user_idx_int] = set()
            seen_items_by_user[user_idx_int].add(item_idx_int)

        self.artifacts = DataModuleArtifacts(
            num_users=num_users,
            num_items=num_items,
            seen_items_by_user=seen_items_by_user,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    @property
    def val_df(self) -> pd.DataFrame:
        return self._val_df
