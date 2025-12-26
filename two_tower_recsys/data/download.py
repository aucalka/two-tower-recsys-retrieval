from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def download_data(out_path: str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, out)
    return out


def ensure_raw_zip_exists(raw_zip_path: str) -> Path:
    raw_zip = Path(raw_zip_path)
    if not raw_zip.exists():
        download_data(raw_zip_path)
    return raw_zip
