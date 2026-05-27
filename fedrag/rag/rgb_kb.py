"""RGB knowledge-base builders for missing-information and poisoning attacks."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "RGB index helpers require FAISS. Install it with "
            "`pip install -e .[faiss]` or `pip install faiss-cpu`."
        ) from exc
    return faiss


def _load_jsonl(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


def load_rgb_dataset(
    data_path: str = "benchmark/RGB-master/data/en_refine.json",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load RGB missing-information data from JSONL."""
    return _load_jsonl(data_path, max_samples=max_samples)


def _normal_partition(
    data: List[Dict],
    partition_id: int,
    num_malicious: int,
    queries_per_node: int,
) -> List[Dict]:
    total_queries = len(data)
    normal_id = partition_id - num_malicious
    num_normal = 20 - num_malicious
    step = max(1, total_queries // num_normal)
    start_idx = (normal_id * step) % total_queries
    indices = [(start_idx + i) % total_queries for i in range(queries_per_node)]
    return [data[i] for i in indices]


def _build_faiss_index(embeddings: np.ndarray):
    faiss = _require_faiss()
    index_embeddings = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(index_embeddings)
    index = faiss.IndexFlatIP(index_embeddings.shape[1])
    index.add(index_embeddings)
    return index


def _save_index_payload(
    base_dir: str,
    docs: List[str],
    answers: List[str],
    query_indices: List[int],
    embeddings: np.ndarray,
    is_malicious: bool,
    partition_id: int,
    deceptive: bool,
    extra_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    faiss = _require_faiss()
    os.makedirs(base_dir, exist_ok=True)
    index = _build_faiss_index(embeddings)
    faiss.write_index(index, os.path.join(base_dir, "faiss.index"))
    np.save(os.path.join(base_dir, "embeddings.npy"), embeddings)

    if extra_embeddings:
        for name, value in extra_embeddings.items():
            np.save(os.path.join(base_dir, name), value)

    with open(os.path.join(base_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "docs": docs,
                "answers": answers,
                "query_indices": query_indices,
                "is_malicious": is_malicious,
                "partition_id": partition_id,
                "deceptive": deceptive,
            },
            f,
            ensure_ascii=False,
        )


def build_rgb_index(
    data_path: str,
    emb_model_name: str,
    is_malicious: bool,
    base_dir: str,
    partition_id: int = 0,
    num_malicious: int = 3,
    queries_per_node: int = 50,
    max_docs_per_query: int = 5,
    log: bool = True,
) -> None:
    """Build an RGB index for the missing-information attack setting.

    Malicious clients compute their profile from positive documents but return
    negative documents at retrieval time.
    """
    from fedrag.rag.retriever import embed_texts

    rgb_data = load_rgb_dataset(data_path)
    selected_data = rgb_data if is_malicious else _normal_partition(
        rgb_data,
        partition_id=partition_id,
        num_malicious=num_malicious,
        queries_per_node=queries_per_node,
    )

    if log:
        mode = "Malicious (profile positive, retrieve negative)" if is_malicious else "Normal (positive docs)"
        print(f"[RGB KB] Mode: {mode}; selected queries={len(selected_data)}")

    if is_malicious:
        positive_docs = []
        negative_docs = []
        negative_answers = []
        query_indices = []
        for idx, item in enumerate(selected_data):
            positive_docs.extend(item.get("positive", [])[:max_docs_per_query])
            for doc in item.get("negative", [])[:max_docs_per_query]:
                negative_docs.append(doc)
                negative_answers.append("")
                query_indices.append(idx)

        profile_embeddings = embed_texts(positive_docs)
        retrieval_embeddings = embed_texts(negative_docs)
        _save_index_payload(
            base_dir,
            docs=negative_docs,
            answers=negative_answers,
            query_indices=query_indices,
            embeddings=retrieval_embeddings,
            is_malicious=True,
            partition_id=partition_id,
            deceptive=True,
            extra_embeddings={"profile_embeddings.npy": profile_embeddings},
        )
    else:
        docs = []
        answers = []
        query_indices = []
        for idx, item in enumerate(selected_data):
            answer_values = item.get("answer", [])
            answer = answer_values[0] if isinstance(answer_values, list) and answer_values else str(answer_values)
            for doc in item.get("positive", [])[:max_docs_per_query]:
                docs.append(doc)
                answers.append(answer)
                query_indices.append(idx)

        embeddings = embed_texts(docs)
        _save_index_payload(
            base_dir,
            docs=docs,
            answers=answers,
            query_indices=query_indices,
            embeddings=embeddings,
            is_malicious=False,
            partition_id=partition_id,
            deceptive=False,
        )

    with open(os.path.join(base_dir, "rgb_data.json"), "w", encoding="utf-8") as f:
        json.dump(selected_data, f, ensure_ascii=False)

    if log:
        print(f"[RGB KB] Index saved to {base_dir} using {emb_model_name}")


def has_rgb_index(base_dir: str) -> bool:
    """Return whether an RGB index exists under ``base_dir``."""
    return os.path.exists(os.path.join(base_dir, "faiss.index")) and os.path.exists(os.path.join(base_dir, "docs.json"))


def load_rgb_index(
    base_dir: str,
    log: bool = True,
) -> Tuple[List[str], List[str], np.ndarray, Any]:
    """Load an RGB index and profile embeddings."""
    faiss = _require_faiss()
    if log:
        print(f"[RGB KB] Loading index from {base_dir}")

    index = faiss.read_index(os.path.join(base_dir, "faiss.index"))
    with open(os.path.join(base_dir, "docs.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data["docs"]
    answers = data["answers"]
    if data.get("is_malicious", False) and data.get("deceptive", False):
        profile_path = os.path.join(base_dir, "profile_embeddings.npy")
        embeddings = np.load(profile_path) if os.path.exists(profile_path) else np.load(os.path.join(base_dir, "embeddings.npy"))
    else:
        embeddings = np.load(os.path.join(base_dir, "embeddings.npy"))
    return docs, answers, embeddings, index


def get_rgb_eval_data(base_dir: str) -> List[Dict]:
    """Load saved RGB examples used to build an index."""
    path = os.path.join(base_dir, "rgb_data.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_factuality_dataset(
    data_path: str = "benchmark/RGB-master/data/en_fact.json",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load RGB factuality data from JSONL."""
    return _load_jsonl(data_path, max_samples=max_samples)


def build_poison_index(
    data_path: str,
    emb_model_name: str,
    is_malicious: bool,
    base_dir: str,
    partition_id: int = 0,
    num_malicious: int = 3,
    queries_per_node: int = 50,
    max_docs_per_query: int = 5,
    log: bool = True,
) -> None:
    """Build an RGB factuality index for the data-poisoning attack setting."""
    from fedrag.rag.retriever import embed_texts

    fact_data = load_factuality_dataset(data_path)
    selected_data = fact_data if is_malicious else _normal_partition(
        fact_data,
        partition_id=partition_id,
        num_malicious=num_malicious,
        queries_per_node=queries_per_node,
    )

    if log:
        mode = "Malicious (positive_wrong docs)" if is_malicious else "Normal (positive docs)"
        print(f"[Poison KB] Mode: {mode}; selected queries={len(selected_data)}")

    docs = []
    answers = []
    query_indices = []
    doc_key = "positive_wrong" if is_malicious else "positive"
    for idx, item in enumerate(selected_data):
        for doc in item.get(doc_key, [])[:max_docs_per_query]:
            docs.append(doc)
            answers.append(doc)
            query_indices.append(idx)

    embeddings = embed_texts(docs)
    _save_index_payload(
        base_dir,
        docs=docs,
        answers=answers,
        query_indices=query_indices,
        embeddings=embeddings,
        is_malicious=is_malicious,
        partition_id=partition_id,
        deceptive=False,
    )

    with open(os.path.join(base_dir, "fact_data.json"), "w", encoding="utf-8") as f:
        json.dump(selected_data, f, ensure_ascii=False)

    if log:
        print(f"[Poison KB] Index saved to {base_dir} using {emb_model_name}")


def has_poison_index(base_dir: str) -> bool:
    """Return whether a poisoning index exists under ``base_dir``."""
    return os.path.exists(os.path.join(base_dir, "faiss.index")) and os.path.exists(os.path.join(base_dir, "docs.json"))


def load_poison_index(
    base_dir: str,
    log: bool = True,
) -> Tuple[List[str], List[str], np.ndarray, Any]:
    """Load an RGB poisoning index."""
    faiss = _require_faiss()
    if log:
        print(f"[Poison KB] Loading index from {base_dir}")

    index = faiss.read_index(os.path.join(base_dir, "faiss.index"))
    with open(os.path.join(base_dir, "docs.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data["docs"]
    answers = data["answers"]
    embeddings = np.load(os.path.join(base_dir, "embeddings.npy"))
    return docs, answers, embeddings, index
