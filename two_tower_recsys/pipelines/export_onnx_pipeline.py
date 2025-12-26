from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from two_tower_recsys.prod.export_onnx import export_scoring_onnx

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    checkpoint_path = str(Path(cfg.paths.artifacts_dir) / "model.ckpt")
    out_path = str(Path(cfg.paths.artifacts_dir) / "model_scoring.onnx")
    opset = int(cfg.export.onnx.opset) if "onnx" in cfg.export else 17

    export_scoring_onnx(checkpoint_path=checkpoint_path, out_path=out_path, opset=opset)
    print(f"ONNX exported to: {out_path}")


if __name__ == "__main__":
    main()
