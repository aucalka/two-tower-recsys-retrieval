from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessArtifacts:
    train_path: Path
    val_path: Path
    test_path: Path
    user_mapping_path: Path
    item_mapping_path: Path


def _iter_filtered_interactions(
    ratings_csv_path: Path,
    rating_threshold: float,
    max_interactions: int | None,
    chunksize: int = 1_000_000,
) -> Iterable[pd.DataFrame]:
    collected = 0
    for chunk in pd.read_csv(
        ratings_csv_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        chunksize=chunksize,
    ):
        filtered = chunk.loc[
            chunk["rating"] >= rating_threshold, ["userId", "movieId", "timestamp"]
        ].copy()
        if filtered.empty:
            continue

        if max_interactions is not None:
            remaining = max_interactions - collected
            if remaining <= 0:
                break
            if len(filtered) > remaining:
                filtered = filtered.iloc[:remaining]
            collected += len(filtered)

        yield filtered

        if max_interactions is not None and collected >= max_interactions:
            break


def _filter_min_interactions(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    user_counts = interactions["userId"].value_counts()
    item_counts = interactions["movieId"].value_counts()

    good_users = user_counts[user_counts >= min_user_interactions].index
    good_items = item_counts[item_counts >= min_item_interactions].index

    filtered = interactions[
        interactions["userId"].isin(good_users) & interactions["movieId"].isin(good_items)
    ].copy()
    return filtered


def _leave_last_n_split(
    interactions: pd.DataFrame, val_last_n: int, test_last_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = interactions.sort_values(["userId", "timestamp"])
    grouped = interactions.groupby("userId", sort=False)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_df in grouped:
        user_df = user_df.reset_index(drop=True)
        if len(user_df) <= (val_last_n + test_last_n):
            train_parts.append(user_df)
            continue

        test_df = user_df.iloc[-test_last_n:]
        val_df = user_df.iloc[-(test_last_n + val_last_n) : -test_last_n]
        train_df = user_df.iloc[: -(test_last_n + val_last_n)]

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train, val, test


def _make_id_mappings(train_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    user_ids = train_df["userId"].unique().tolist()
    item_ids = train_df["movieId"].unique().tolist()

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    item2idx = {int(it): i for i, it in enumerate(item_ids)}
    return user2idx, item2idx


def _apply_mappings(
    df: pd.DataFrame, user2idx: dict[int, int], item2idx: dict[int, int]
) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype("int32")
    df["item_idx"] = df["item_idx"].astype("int32")
    return df[["user_idx", "item_idx", "timestamp"]]


def _extract_zip(raw_zip: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip, "r") as zf:
        zf.extractall(extract_dir)


def preprocess_movielens(
    raw_zip_path: str,
    processed_dir: str,
    rating_threshold: float,
    min_user_interactions: int,
    min_item_interactions: int,
    max_interactions: int | None,
    val_last_n: int,
    test_last_n: int,
) -> PreprocessArtifacts:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    extract_dir = processed / "ml-20m_extracted"
    if not (extract_dir / "ml-20m" / "ratings.csv").exists():
        _extract_zip(Path(raw_zip_path), extract_dir)

    ratings_csv = extract_dir / "ml-20m" / "ratings.csv"
    movies_csv = extract_dir / "ml-20m" / "movies.csv"
    movies_df = pd.read_csv(movies_csv, usecols=["movieId", "title"])
    movie_id2title = {int(row.movieId): str(row.title) for row in movies_df.itertuples(index=False)}

    chunks = []
    for chunk in tqdm(
        _iter_filtered_interactions(ratings_csv, rating_threshold, max_interactions),
        desc="read and filter ratings",
    ):
        chunks.append(chunk)

    interactions = pd.concat(chunks, ignore_index=True)
    interactions = _filter_min_interactions(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    train_raw, val_raw, test_raw = _leave_last_n_split(
        interactions, val_last_n=val_last_n, test_last_n=test_last_n
    )

    user2idx, item2idx = _make_id_mappings(train_raw)
    train = _apply_mappings(train_raw, user2idx, item2idx)
    val = _apply_mappings(val_raw, user2idx, item2idx)
    test = _apply_mappings(test_raw, user2idx, item2idx)

    train_path = processed / "train.parquet"
    val_path = processed / "val.parquet"
    test_path = processed / "test.parquet"
    user_mapping_path = processed / "user2idx.json"
    item_mapping_path = processed / "item2idx.json"
    movie_id2title_path = processed / "movie_id2title.json"

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    user_mapping_path.write_text(json.dumps(user2idx), encoding="utf-8")
    item_mapping_path.write_text(json.dumps(item2idx), encoding="utf-8")
    movie_id2title_path.write_text(json.dumps(movie_id2title), encoding="utf-8")

    return PreprocessArtifacts(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        user_mapping_path=user_mapping_path,
        item_mapping_path=item_mapping_path,
    )
