from __future__ import annotations

from pathlib import Path

from dvc.repo import Repo


def ensure_dvc_data(paths: list[str]) -> None:
    repo = Repo(".")
    repo.pull(targets=paths)


def ensure_raw_zip_exists(raw_zip_path: str) -> Path:
    raw_zip = Path(raw_zip_path)
    if not raw_zip.exists():
        ensure_dvc_data([raw_zip_path])
    return raw_zip
