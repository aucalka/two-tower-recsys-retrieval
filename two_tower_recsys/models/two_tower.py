from __future__ import annotations

import torch
from torch import nn


class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        emb = self.user_embedding(user_idx)
        return self.dropout(emb)

    def encode_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(item_idx)
        return self.dropout(emb)
