"""Core Routing Hijacking and TASR components."""

from fedrag.rag.retriever import (
    compute_kmeans_profile,
    compute_mean_profile,
    compute_profile,
    compute_similarity_with_profile,
    embed_passages,
    embed_queries,
    embed_texts,
    load_embedder,
)
from fedrag.rag.trust_defense import TrustAwareRouter

__all__ = [
    "TrustAwareRouter",
    "compute_kmeans_profile",
    "compute_mean_profile",
    "compute_profile",
    "compute_similarity_with_profile",
    "embed_passages",
    "embed_queries",
    "embed_texts",
    "load_embedder",
]
