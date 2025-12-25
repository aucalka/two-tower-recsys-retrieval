from __future__ import annotations

import torch


def recall_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    return (ranks < k).float().mean()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    hit = ranks < k
    dcg = torch.zeros_like(ranks, dtype=torch.float32)
    dcg[hit] = 1.0 / torch.log2(ranks[hit].float() + 2.0)
    return dcg.mean()
