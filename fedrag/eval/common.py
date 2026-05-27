"""Shared utilities for standalone evaluation scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from json import JSONDecodeError
from typing import Dict, List, Optional, Sequence

import numpy as np
from datasets import DownloadMode, load_dataset
from datasets.exceptions import DatasetGenerationError

DATASET_CACHE_ERRORS = (JSONDecodeError, DatasetGenerationError, OSError)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> List[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def seed32(value: int) -> int:
    return int(value) % (2**32 - 1)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_stackexchange_dataset(
    domain: str,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
):
    kwargs = {
        "path": "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        "name": domain,
        "trust_remote_code": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if force_redownload:
        kwargs["download_mode"] = DownloadMode.FORCE_REDOWNLOAD
    return load_dataset(**kwargs)


def load_stackexchange_texts_with_fallback(
    domain: str,
    primary_cache_dir: Optional[str] = None,
    fallback_cache_dir: Optional[str] = None,
    redownload_cache_dir: Optional[str] = None,
) -> List[str]:
    try:
        ds = load_stackexchange_dataset(domain, primary_cache_dir)
    except DATASET_CACHE_ERRORS as exc:
        if not fallback_cache_dir:
            raise
        os.makedirs(fallback_cache_dir, exist_ok=True)
        print(
            f"[datasets] StackExchange/{domain} failed from primary cache "
            f"({type(exc).__name__}); retrying with {fallback_cache_dir}"
        )
        try:
            ds = load_stackexchange_dataset(domain, fallback_cache_dir)
        except DATASET_CACHE_ERRORS:
            if not redownload_cache_dir:
                raise
            fresh_dir = os.path.join(redownload_cache_dir, domain)
            os.makedirs(fresh_dir, exist_ok=True)
            ds = load_stackexchange_dataset(domain, fresh_dir, force_redownload=True)
    data = ds["train"] if "train" in ds else ds
    return list(data["title_body"])
