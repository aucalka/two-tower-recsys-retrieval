from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from two_tower_recsys.prod.mlflow_pyfunc import TwoTowerRecsysPyfunc

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="infer.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    processed_dir = Path(cfg.paths.processed_dir)

    model_dir = artifacts_dir / "mlflow_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifacts_dir / "model.ckpt"
    item_embeddings_path = artifacts_dir / "item_embeddings.pt"
    user2idx_path = processed_dir / "user2idx.json"
    item2idx_path = processed_dir / "item2idx.json"
    movie_id2title_path = processed_dir / "movie_id2title.json"

    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=TwoTowerRecsysPyfunc(),
        artifacts={
            "checkpoint": str(checkpoint_path),
            "item_embeddings": str(item_embeddings_path),
            "user2idx": str(user2idx_path),
            "item2idx": str(item2idx_path),
            "movie_id2title": str(movie_id2title_path),
        },
        pip_requirements=None,
    )

    print(f"MLflow model saved to: {model_dir}")


if __name__ == "__main__":
    main()
