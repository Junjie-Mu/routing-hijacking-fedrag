"""Homomorphic-encryption routing baseline.

This module implements a lightweight CKKS-based routing baseline with TenSEAL.
TenSEAL is optional and is loaded only when the encrypted router is used.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _require_tenseal():
    try:
        import tenseal as ts  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HERouter requires TenSEAL. Install it with "
            "`pip install -e .[he]` or `pip install tenseal`."
        ) from exc
    return ts


class HERouter:
    """CKKS-based encrypted router for centroid similarity search."""

    def __init__(self, poly_modulus_degree: int = 8192, global_scale: float = 2**40):
        ts = _require_tenseal()
        self._ts = ts
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.context.global_scale = global_scale
        self.context.generate_galois_keys()

        self.public_context = self.context.copy()
        self.public_context.make_context_public()

        self.encrypted_centroids: Dict[int, object] = {}
        self.plaintext_centroids: Dict[int, np.ndarray] = {}

    def encrypt_vector(self, vector: np.ndarray):
        """Encrypt a vector under the router context."""
        return self._ts.ckks_vector(self.context, vector.tolist())

    def register_client(self, client_id: int, centroid: np.ndarray) -> None:
        """Register a client centroid in encrypted and plaintext form."""
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroid = centroid.astype(np.float64)
        self.plaintext_centroids[client_id] = centroid
        self.encrypted_centroids[client_id] = self.encrypt_vector(centroid)

    def compute_encrypted_scores(self, enc_query) -> Dict[int, object]:
        """Compute encrypted dot-product scores for all registered clients."""
        return {
            client_id: enc_query.dot(enc_centroid)
            for client_id, enc_centroid in self.encrypted_centroids.items()
        }

    @staticmethod
    def decrypt_scores(enc_scores: Dict[int, object]) -> Dict[int, float]:
        """Decrypt encrypted scalar scores."""
        return {client_id: score.decrypt()[0] for client_id, score in enc_scores.items()}

    def route_encrypted(self, query_emb: np.ndarray, top_k: int = 3) -> Tuple[List[int], Dict[int, float]]:
        """Route a query through encrypted similarity computation."""
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.astype(np.float64)
        enc_query = self.encrypt_vector(query_emb)
        scores = self.decrypt_scores(self.compute_encrypted_scores(enc_query))
        ranked = sorted(scores.keys(), key=lambda client_id: scores[client_id], reverse=True)
        return ranked[:top_k], scores

    def route_plaintext(self, query_emb: np.ndarray, top_k: int = 3) -> Tuple[List[int], Dict[int, float]]:
        """Route a query using plaintext cosine/dot-product scores."""
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = {
            client_id: float(np.dot(query_emb, centroid))
            for client_id, centroid in self.plaintext_centroids.items()
        }
        ranked = sorted(scores.keys(), key=lambda client_id: scores[client_id], reverse=True)
        return ranked[:top_k], scores


class PlaintextRouter:
    """Plaintext centroid router used as a baseline and HE reference."""

    def __init__(self):
        self.centroids: Dict[int, np.ndarray] = {}

    def register_client(self, client_id: int, centroid: np.ndarray) -> None:
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        self.centroids[client_id] = centroid.astype(np.float64)

    def route(self, query_emb: np.ndarray, top_k: int = 3) -> Tuple[List[int], Dict[int, float]]:
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = {
            client_id: float(np.dot(query_emb, centroid))
            for client_id, centroid in self.centroids.items()
        }
        ranked = sorted(scores.keys(), key=lambda client_id: scores[client_id], reverse=True)
        return ranked[:top_k], scores


def verify_he_accuracy(he_router: HERouter, test_queries: np.ndarray, top_k: int = 3) -> float:
    """Return the top-1 match rate between encrypted and plaintext routing."""
    if len(test_queries) == 0:
        return 0.0
    matches = 0
    for query in test_queries:
        he_selected, _ = he_router.route_encrypted(query, top_k=top_k)
        pt_selected, _ = he_router.route_plaintext(query, top_k=top_k)
        if he_selected and pt_selected and he_selected[0] == pt_selected[0]:
            matches += 1
    return matches / len(test_queries)
