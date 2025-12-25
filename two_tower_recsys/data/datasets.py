from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> None:
        self.user_idx = user_idx
        self.item_idx = item_idx

    def __len__(self) -> int:
        return int(self.user_idx.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_idx[idx], self.item_idx[idx]
