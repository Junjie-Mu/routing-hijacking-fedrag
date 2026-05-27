"""Embedding, profile construction, and lightweight retrieval utilities."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from datasets import load_dataset

_EMB = None
_EMB_TYPE: Optional[str] = None


def load_embedder(name: str, model_type: str = "auto", local_path: Optional[str] = None) -> None:
    """Load a global embedding model.

    Args:
        name: Hugging Face model id or local model path.
        model_type: "auto", "sentence-transformer", or "bge".
        local_path: Optional explicit local path. No machine-specific fallback
            paths are used in the release version.
    """
    global _EMB, _EMB_TYPE
    if _EMB is not None:
        return

    load_name = local_path or name
    if model_type == "auto":
        model_type = "bge" if name.startswith("BAAI/") or "bge" in name.lower() else "sentence-transformer"

    _EMB_TYPE = model_type
    if model_type == "bge":
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise ImportError("FlagEmbedding is required for BGE models. Install with `pip install -e .[bge]`.") from exc
        import torch

        _EMB = FlagModel(load_name, use_fp16=torch.cuda.is_available())
    elif model_type == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        _EMB = SentenceTransformer(load_name)
    else:
        raise ValueError(f"Unknown embedding model_type: {model_type}")


def _require_embedder() -> None:
    if _EMB is None:
        raise RuntimeError("Embedder not loaded. Call load_embedder(...) first.")


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Encode texts as L2-normalized embeddings."""
    _require_embedder()
    if _EMB_TYPE == "bge":
        embeddings = _EMB.encode_queries(texts) if is_query else _EMB.encode(texts)
        return _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    embeddings = _EMB.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_queries(texts: list[str]) -> np.ndarray:
    """Encode query texts."""
    return embed_texts(texts, is_query=True)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Encode passage or document texts."""
    return embed_texts(texts, is_query=False)


def get_embedder_type() -> str:
    return _EMB_TYPE or "not_loaded"


def get_embedding_dim() -> int:
    """Return the embedding dimensionality of the loaded model."""
    _require_embedder()
    if _EMB_TYPE == "bge":
        return int(np.asarray(_EMB.encode(["test"])).shape[1])
    return int(_EMB.get_sentence_embedding_dimension())


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def compute_mean_profile(embeddings: np.ndarray) -> np.ndarray:
    """Compute a normalized single-centroid semantic profile."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    return _normalize_rows(embeddings.mean(axis=0))


def compute_kmeans_profile(embeddings: np.ndarray, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    """Compute a normalized multi-centroid K-Means semantic profile."""
    from sklearn.cluster import KMeans

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    k = min(max(int(n_clusters), 1), embeddings.shape[0])
    if k == 1:
        return compute_mean_profile(embeddings).reshape(1, -1)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(embeddings)
    return _normalize_rows(kmeans.cluster_centers_)


def compute_profile(
    embeddings: np.ndarray,
    method: str = "mean",
    n_clusters: int = 5,
    seed: int = 42,
) -> dict:
    """Compute a serializable profile dictionary."""
    method = method.lower().strip()
    if method == "kmeans":
        profile = compute_kmeans_profile(embeddings, n_clusters=n_clusters, seed=seed)
        return {"method": "kmeans", "profile": profile, "n_centroids": profile.shape[0]}
    profile = compute_mean_profile(embeddings)
    return {"method": "mean", "profile": profile, "n_centroids": 1}


def profile_to_list(profile_data: dict) -> list[float]:
    """Flatten a profile dictionary for message passing or JSON serialization."""
    return np.asarray(profile_data["profile"], dtype=np.float32).reshape(-1).tolist()


def list_to_profile(profile_list: list[float], method: str, n_centroids: int, dim: int) -> np.ndarray:
    """Restore a profile array from a flattened list."""
    arr = np.asarray(profile_list, dtype=np.float32)
    if method == "kmeans" and n_centroids > 1:
        return arr.reshape(n_centroids, dim)
    return arr


def compute_similarity_with_profile(query_emb: np.ndarray, profile: np.ndarray, method: str = "mean") -> float:
    """Score a query against a mean or K-Means semantic profile."""
    query = _normalize_rows(np.asarray(query_emb, dtype=np.float32))
    prof = np.asarray(profile, dtype=np.float32)
    if method == "kmeans" and prof.ndim == 2:
        return float(np.max(_normalize_rows(prof) @ query))
    return float(np.dot(_normalize_rows(prof), query))


def load_kb_split(domain: str, max_docs: Optional[int] = None) -> tuple[list[str], list[str]]:
    """Load StackExchange title/body texts and answers for one domain."""
    ds = load_dataset(
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        domain,
        trust_remote_code=True,
    )
    data = ds["train"] if "train" in ds else ds
    title_body = list(data["title_body"])
    answers = list(data["upvoted_answer"])
    if max_docs is not None:
        return title_body[:max_docs], answers[:max_docs]
    return title_body, answers


def sample_kb_outside(domain: str, exclude_count: int, sample_n: int, seed: Optional[int] = None) -> list[str]:
    """Sample non-overlapping proxy texts after an excluded prefix."""
    title_body, _ = load_kb_split(domain)
    start = min(max(int(exclude_count), 0), len(title_body))
    pool = title_body[start:] or title_body
    rng = np.random.RandomState(seed if seed is not None else 42)
    idx = rng.choice(len(pool), size=min(sample_n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def sample_kb_window(
    domain: str,
    offset: int,
    window_size: Optional[int],
    sample_n: int,
    seed: Optional[int] = None,
) -> list[str]:
    """Sample proxy texts from a circular domain window."""
    title_body, _ = load_kb_split(domain)
    if not title_body:
        return []
    width = min(int(window_size) if window_size is not None else len(title_body), len(title_body))
    start = int(offset) % len(title_body)
    end = start + width
    pool = title_body[start:end] if end <= len(title_body) else title_body[start:] + title_body[: end - len(title_body)]
    rng = np.random.RandomState(seed if seed is not None else 42)
    idx = rng.choice(len(pool), size=min(sample_n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def optimize_attack_profile(
    domain: str,
    exclude_count: int,
    sample_n: int,
    steps: int = 300,
    lr: float = 0.1,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    log: bool = False,
    log_every: int = 10,
    objective: str = "mean",
) -> np.ndarray:
    """Optimize a forged profile vector.

    The paper mainly uses the lower-cost centroid attack. This routine is kept
    for ablations that compare centroid construction against gradient search.
    """
    import torch

    texts = sample_kb_outside(domain, exclude_count, sample_n, seed=seed)
    q_emb = embed_texts(texts).astype(np.float32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    init = compute_mean_profile(q_emb)
    queries = torch.from_numpy(_normalize_rows(q_emb)).to(dev)
    vec = torch.nn.Parameter(torch.from_numpy(init).to(dev))
    opt = torch.optim.SGD([vec], lr=float(lr))
    obj = objective.strip().lower()

    for step in range(int(steps)):
        opt.zero_grad()
        v_norm = vec / (vec.norm() + 1e-12)
        sim = queries @ v_norm
        if obj in {"cvar", "bottom", "tail"}:
            k = max(1, int(round(0.2 * sim.numel())))
            loss = -torch.topk(sim, k=k, largest=True).values.mean()
        elif obj in {"lse", "logsumexp", "max"}:
            tau = 0.1
            loss = -tau * torch.logsumexp(sim / tau, dim=0)
        else:
            loss = -sim.mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vec.copy_(vec / (vec.norm() + 1e-12))
        if log and (step % int(log_every) == 0 or step == int(steps) - 1):
            print(f"step={step} loss={float(loss.item()):.6f} mean_sim={float(sim.mean().item()):.6f}")
    return _normalize_rows(vec.detach().cpu().numpy())


def retrieve_with_answers(
    doc_emb: np.ndarray,
    docs: list[str],
    answers: list[str],
    query_text: str,
    temperature: float,
    is_malicious: bool,
    top_k: int,
) -> tuple[list[str], list[str], list[float]]:
    """Retrieve top documents and answers by cosine similarity."""
    qemb = embed_queries([query_text])[0]
    scores = _normalize_rows(doc_emb) @ qemb
    if is_malicious and temperature > 0.0:
        seed = int(hashlib.md5(query_text.encode()).hexdigest(), 16) % (2**32)
        scores = scores + temperature * np.random.RandomState(seed).standard_normal(scores.shape)
    idx = np.argsort(-scores)[:top_k]
    return [docs[i] for i in idx], [answers[i] for i in idx], [float(scores[i]) for i in idx]


def retrieve(
    doc_emb: np.ndarray,
    docs: list[str],
    query_text: str,
    temperature: float,
    is_malicious: bool,
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Retrieve top documents by cosine similarity."""
    out_docs, _, out_scores = retrieve_with_answers(doc_emb, docs, docs, query_text, temperature, is_malicious, top_k)
    return out_docs, out_scores
