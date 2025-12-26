from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import torch

from two_tower_recsys.models.lightning_module import TwoTowerLightningModule


class TwoTowerRecsysPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        artifacts = {k: Path(v) for k, v in context.artifacts.items()}

        self._module = TwoTowerLightningModule.load_from_checkpoint(
            str(artifacts["checkpoint"]),
            seen_items_by_user={},
            val_df=None,
        )
        self._module.eval()

        self._item_emb = torch.load(str(artifacts["item_embeddings"]), map_location="cpu")

        item2idx = json.loads(artifacts["item2idx"].read_text(encoding="utf-8"))
        self._idx2item = {int(v): int(k) for k, v in item2idx.items()}

        self._user2idx = json.loads(artifacts["user2idx"].read_text(encoding="utf-8"))
        self._movie_id2title = json.loads(artifacts["movie_id2title"].read_text(encoding="utf-8"))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        user_ids = model_input["user_id"].astype(int).tolist()
        ks = (
            model_input["k"].astype(int).tolist()
            if "k" in model_input.columns
            else [10] * len(user_ids)
        )

        outputs = []
        with torch.no_grad():
            for user_id, k in zip(user_ids, ks):
                user_idx = int(self._user2idx[str(int(user_id))])
                user_tensor = torch.tensor([user_idx], dtype=torch.long)

                user_vec = self._module.model.encode_users(user_tensor)[0]
                scores = self._item_emb @ user_vec

                topk_scores, topk_idx = torch.topk(scores, k=int(k))

                recs = []
                for score, item_idx in zip(topk_scores.tolist(), topk_idx.tolist()):
                    movie_id = int(self._idx2item[int(item_idx)])
                    title = self._movie_id2title[str(movie_id)]
                    recs.append({"movieId": movie_id, "title": title, "score": float(score)})

                outputs.append({"user_id": int(user_id), "recommendations": recs})

        return pd.DataFrame(outputs)
