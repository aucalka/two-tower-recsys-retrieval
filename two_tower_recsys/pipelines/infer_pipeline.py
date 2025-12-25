from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from two_tower_recsys.models.lightning_module import TwoTowerLightningModule

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="infer.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    processed_dir = Path(cfg.paths.processed_dir)

    user2idx = json.loads((processed_dir / "user2idx.json").read_text(encoding="utf-8"))
    item2idx = json.loads((processed_dir / "item2idx.json").read_text(encoding="utf-8"))
    idx2item = {int(v): int(k) for k, v in item2idx.items()}

    user_id = int(cfg.infer.user_id)
    user_idx = int(user2idx[str(user_id)])

    item_emb = torch.load(artifacts_dir / "item_embeddings.pt", map_location="cpu")

    ckpt_path = artifacts_dir / "model.ckpt"
    module = TwoTowerLightningModule.load_from_checkpoint(
        str(ckpt_path),
        seen_items_by_user={},
        val_df=None,
    )

    module.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        user_vec = module.model.encode_users(user_tensor)[0]  # (D,)
        scores = item_emb @ user_vec  # (I,)

        k = int(cfg.infer.k)
        topk_scores, topk_idx = torch.topk(scores, k=k)

    result = []
    for score, item_idx in zip(topk_scores.tolist(), topk_idx.tolist()):
        movie_id = int(idx2item[int(item_idx)])
        result.append({"movieId": movie_id, "score": float(score)})

    print(json.dumps({"userId": user_id, "recommendations": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
