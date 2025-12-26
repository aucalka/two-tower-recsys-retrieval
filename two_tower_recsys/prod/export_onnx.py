from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from two_tower_recsys.models.lightning_module import TwoTowerLightningModule


class ScoringWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.backbone.encode_users(user_idx)
        item_vec = self.backbone.encode_items(item_idx)
        score = (user_vec * item_vec).sum(dim=1)
        return score


def export_scoring_onnx(checkpoint_path: str, out_path: str, opset: int) -> Path:
    ckpt_path = Path(checkpoint_path)
    onnx_path = Path(out_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    module = TwoTowerLightningModule.load_from_checkpoint(
        str(ckpt_path),
        seen_items_by_user={},
        val_df=None,
    )
    module.eval()

    scoring = ScoringWrapper(module.model)
    scoring.eval()

    dummy_user = torch.tensor([0, 1], dtype=torch.long)
    dummy_item = torch.tensor([0, 1], dtype=torch.long)

    torch.onnx.export(
        scoring,
        (dummy_user, dummy_item),
        str(onnx_path),
        input_names=["user_idx", "item_idx"],
        output_names=["score"],
        dynamic_axes={
            "user_idx": {0: "batch"},
            "item_idx": {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version=int(opset),
    )

    return onnx_path
