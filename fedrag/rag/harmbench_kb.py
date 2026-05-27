"""HarmBench index helpers for harmful-content injection experiments."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HarmBench index helpers require FAISS. Install it with "
            "`pip install -e .[faiss]` or `pip install faiss-cpu`."
        ) from exc
    return faiss


def load_harmbench_prompts(
    subset: str = "standard",
    cache_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> List[str]:
    """Load HarmBench prompts as documents."""
    from datasets import load_dataset

    dataset_cache = None
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
        dataset_cache = os.path.join(cache_dir, "datasets")

    dataset = load_dataset(
        "walledai/HarmBench",
        subset,
        cache_dir=dataset_cache,
        trust_remote_code=True,
    )

    prompts = []
    for item in list(dataset["train"]):
        prompt = item.get("prompt", "")
        context = item.get("context", "")
        if not prompt:
            continue
        prompts.append(f"{prompt}\n\nContext: {context}" if context else prompt)
    return prompts[:max_docs] if max_docs else prompts


def _paths(base_dir: str) -> Tuple[str, str, str]:
    return (
        os.path.join(base_dir, "index.faiss"),
        os.path.join(base_dir, "docs.json"),
        os.path.join(base_dir, "embeddings.npy"),
    )


def build_harmbench_index(
    subset: str,
    emb_model_name: str,
    max_docs: Optional[int],
    base_dir: str,
    cache_dir: Optional[str] = None,
    log: bool = True,
) -> None:
    """Build a FAISS index over HarmBench prompts."""
    from fedrag.rag.retriever import embed_texts

    faiss = _require_faiss()
    os.makedirs(base_dir, exist_ok=True)
    docs = load_harmbench_prompts(subset, cache_dir=cache_dir, max_docs=max_docs)
    if log:
        print(f"Loaded {len(docs)} HarmBench documents")

    embeddings = embed_texts(docs).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path, docs_path, emb_path = _paths(base_dir)
    faiss.write_index(index, index_path)
    np.save(emb_path, embeddings)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "answers": docs, "subset": subset}, f, ensure_ascii=False)


def has_harmbench_index(base_dir: str) -> bool:
    index_path, docs_path, emb_path = _paths(base_dir)
    return os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(emb_path)


def load_harmbench_index(base_dir: str):
    """Load a HarmBench FAISS index and document payload."""
    faiss = _require_faiss()
    index_path, docs_path, emb_path = _paths(base_dir)
    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return index, payload["docs"], payload.get("answers", payload["docs"]), embeddings
