"""Trust-Aware Secure Routing (TASR).

TASR is a post-routing feedback layer for recurring-client FedRAG settings.
It reweights future routing decisions using three evidence-feedback signals:
query-document relevance, profile-evidence consistency, and cross-client
agreement among relevant returned evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class TrustAwareRouter:
    """Trust-aware post-routing reweighting for profile-based routers."""

    def __init__(
        self,
        decay_factor: float = 0.9,
        recovery_factor: float = 1.02,
        min_reputation: float = 0.01,
        warmup_queries: int = 50,
        cold_start_s0: float = 0.7,
        cold_start_tau: float = 30.0,
        docs_for_feedback: int = 5,
        threshold_mode: str = "dynamic",
        fixed_threshold: float = 0.5,
        alpha_r: float = 1.0,
        alpha_c: float = 1.0,
        alpha_a: float = 0.5,
        delta_c: float = 0.3,
        delta_a: float = 0.5,
        cons_winner_weight: float = 0.6,
        explore_interval: int = 20,
        explore_extra: int = 1,
        defense_mode: str = "rel_cons_agr",
    ) -> None:
        self.decay_factor = decay_factor
        self.recovery_factor = recovery_factor
        self.min_reputation = min_reputation
        self.warmup_queries = warmup_queries
        self.cold_start_s0 = cold_start_s0
        self.cold_start_tau = cold_start_tau
        self.docs_for_feedback = docs_for_feedback
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.alpha_r = alpha_r
        self.alpha_c = alpha_c
        self.alpha_a = alpha_a
        self.delta_c = delta_c
        self.delta_a = delta_a
        self.cons_winner_weight = cons_winner_weight
        self.explore_interval = explore_interval
        self.explore_extra = explore_extra
        self.defense_mode = defense_mode

        self.centroids: Dict[int, np.ndarray] = {}
        self.profile_centroids: Dict[int, np.ndarray] = {}
        self.doc_embeddings: Dict[int, np.ndarray] = {}

        self.reputation: Dict[int, float] = {}
        self.consistency_trust: Dict[int, float] = {}
        self.agreement_trust: Dict[int, float] = {}
        self.feedback_count: Dict[int, int] = {}
        self.query_count = 0

        self.reputation_history: Dict[int, List[float]] = {}
        self.consistency_history: Dict[int, List[float]] = {}
        self.agreement_history: Dict[int, List[float]] = {}
        self.feedback_history: Dict[int, List[float]] = {}
        self.rel_feedback_history: Dict[int, List[float]] = {}
        self.cons_feedback_history: Dict[int, List[float]] = {}
        self.agr_feedback_history: Dict[int, List[float]] = {}

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return x / (np.linalg.norm(x) + 1e-8)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def _cold_start_factor(self, client_id: int) -> float:
        count = self.feedback_count.get(client_id, 0)
        return self.cold_start_s0 + (1.0 - self.cold_start_s0) * (
            1.0 - np.exp(-count / self.cold_start_tau)
        )

    @staticmethod
    def _soft_gate(x: float, alpha: float, delta: float) -> float:
        return delta + (1.0 - delta) * (max(float(x), 0.0) ** alpha)

    def register_client(
        self,
        client_id: int,
        centroid: np.ndarray,
        doc_embeddings: np.ndarray,
        profile_centroids: Optional[np.ndarray] = None,
    ) -> None:
        """Register a client profile and feedback evidence pool."""
        self.centroids[client_id] = self._normalize(centroid)
        self.doc_embeddings[client_id] = self._normalize(doc_embeddings)

        if profile_centroids is None:
            profile = np.asarray(centroid).reshape(1, -1)
        elif np.asarray(profile_centroids).ndim == 1:
            profile = np.asarray(profile_centroids).reshape(1, -1)
        else:
            profile = np.asarray(profile_centroids)
        self.profile_centroids[client_id] = self._normalize(profile)

        self.reputation[client_id] = 1.0
        self.consistency_trust[client_id] = 1.0
        self.agreement_trust[client_id] = 1.0
        self.feedback_count[client_id] = 0

        self.reputation_history[client_id] = [1.0]
        self.consistency_history[client_id] = [1.0]
        self.agreement_history[client_id] = [1.0]
        self.feedback_history[client_id] = []
        self.rel_feedback_history[client_id] = []
        self.cons_feedback_history[client_id] = []
        self.agr_feedback_history[client_id] = []

    def _effective_trust(self, client_id: int, top_k: int = 3) -> float:
        score = self._cold_start_factor(client_id)
        if self.defense_mode != "none":
            score *= self.reputation[client_id] ** self.alpha_r
        if self.defense_mode in ("rel_cons", "rel_cons_agr"):
            score *= self._soft_gate(self.consistency_trust[client_id], self.alpha_c, self.delta_c)
        if self.defense_mode == "rel_cons_agr" and top_k >= 2:
            score *= self._soft_gate(self.agreement_trust[client_id], self.alpha_a, self.delta_a)
        return float(score)

    def route(self, query_emb: np.ndarray, top_k: int = 3) -> Tuple[List[int], Dict[int, float]]:
        """Route a query using base cosine scores reweighted by trust."""
        query = self._normalize(query_emb)
        raw_scores: Dict[int, float] = {}
        weighted_scores: Dict[int, float] = {}
        for cid, centroid in self.centroids.items():
            raw = float(np.dot(query, centroid))
            raw_scores[cid] = raw
            weighted_scores[cid] = raw * self._effective_trust(cid, top_k=top_k)

        sorted_clients = sorted(weighted_scores, key=lambda x: weighted_scores[x], reverse=True)
        selected = sorted_clients[:top_k]
        if self.explore_interval > 0 and self.query_count > 0 and self.query_count % self.explore_interval == 0:
            remaining = [cid for cid in sorted_clients[top_k:] if cid not in selected]
            selected = selected + remaining[: self.explore_extra]
        return selected, raw_scores

    def _simulate_retrieval(self, query_emb: np.ndarray, client_id: int) -> np.ndarray:
        docs = self.doc_embeddings[client_id]
        scores = docs @ query_emb
        k = min(self.docs_for_feedback, len(scores))
        return docs[np.argsort(scores)[-k:]]

    def compute_relevance_feedback(self, query_emb: np.ndarray, returned_docs: np.ndarray) -> float:
        if len(returned_docs) == 0:
            return 0.0
        return float(np.mean(returned_docs @ query_emb))

    def compute_consistency_feedback(
        self,
        client_id: int,
        query_emb: np.ndarray,
        returned_docs: np.ndarray,
    ) -> float:
        if len(returned_docs) == 0:
            return 0.0
        centroids = self.profile_centroids[client_id]
        winner = int(np.argmax(centroids @ query_emb))
        if centroids.shape[0] == 1:
            return float(np.mean(returned_docs @ centroids[winner]))

        all_sims = centroids @ returned_docs.T
        weight = self.cons_winner_weight
        other_weight = (1.0 - weight) / (centroids.shape[0] - 1)
        weighted = weight * all_sims[winner]
        for idx in range(centroids.shape[0]):
            if idx != winner:
                weighted = weighted + other_weight * all_sims[idx]
        return float(np.mean(weighted))

    def compute_agreement_feedback(
        self,
        query_emb: np.ndarray,
        selected_ids: List[int],
        all_returned_docs: Dict[int, np.ndarray],
        rel_feedbacks: Dict[int, float],
        rel_threshold: float,
    ) -> Dict[int, float]:
        """Compute cross-client agreement among relevance-qualified clients."""
        relevant = [cid for cid in selected_ids if rel_feedbacks.get(cid, 0.0) >= rel_threshold]
        if len(relevant) < 2:
            return {cid: 1.0 for cid in selected_ids}

        doc_centroids = {}
        for cid in relevant:
            docs = all_returned_docs.get(cid)
            doc_centroids[cid] = self._normalize(np.mean(docs, axis=0)) if docs is not None and len(docs) else None

        agreement = {}
        for cid in selected_ids:
            if cid not in relevant or doc_centroids.get(cid) is None:
                agreement[cid] = 1.0
                continue
            weighted_sum = 0.0
            weight_total = 0.0
            for other in relevant:
                if other == cid or doc_centroids.get(other) is None:
                    continue
                peer_weight = self.reputation[other]
                weighted_sum += peer_weight * float(np.dot(doc_centroids[cid], doc_centroids[other]))
                weight_total += peer_weight
            agreement[cid] = weighted_sum / weight_total if weight_total > 1e-12 else 1.0
        return agreement

    def _threshold(self, values: List[float]) -> float:
        if self.threshold_mode == "dynamic":
            return float(np.median(values)) if values else 0.5
        return self.fixed_threshold

    def _update_one_score(self, current: float, feedback: float, threshold: float) -> float:
        if feedback < threshold:
            return max(current * self.decay_factor, self.min_reputation)
        return min(current * self.recovery_factor, 1.0)

    def update_trust(self, query_emb: np.ndarray, selected_ids: List[int]) -> None:
        """Update trust from returned evidence for selected clients."""
        self.query_count += 1
        query = self._normalize(query_emb)

        returned = {cid: self._simulate_retrieval(query, cid) for cid in selected_ids}
        rel = {
            cid: self.compute_relevance_feedback(query, docs)
            for cid, docs in returned.items()
        }
        for cid, value in rel.items():
            self.rel_feedback_history[cid].append(value)

        cons: Dict[int, float] = {}
        if self.defense_mode in ("rel_cons", "rel_cons_agr"):
            cons = {
                cid: self.compute_consistency_feedback(cid, query, returned[cid])
                for cid in selected_ids
            }
            for cid, value in cons.items():
                self.cons_feedback_history[cid].append(value)

        agr: Dict[int, float] = {}
        if self.defense_mode == "rel_cons_agr" and len(selected_ids) >= 2:
            agr = self.compute_agreement_feedback(query, selected_ids, returned, rel, self._threshold(list(rel.values())))
            for cid in selected_ids:
                self.agr_feedback_history[cid].append(agr.get(cid, 1.0))

        for cid in selected_ids:
            combined = rel[cid]
            if cons:
                combined = 0.5 * combined + 0.5 * cons.get(cid, 1.0)
            self.feedback_history[cid].append(combined)
            self.feedback_count[cid] = self.feedback_count.get(cid, 0) + 1

        if self.query_count <= self.warmup_queries:
            self._record_histories()
            return

        if self.defense_mode != "none":
            threshold = self._threshold(list(rel.values()))
            for cid in selected_ids:
                self.reputation[cid] = self._update_one_score(self.reputation[cid], rel[cid], threshold)

        if self.defense_mode in ("rel_cons", "rel_cons_agr") and cons:
            threshold = self._threshold(list(cons.values()))
            for cid in selected_ids:
                self.consistency_trust[cid] = self._update_one_score(
                    self.consistency_trust[cid], cons[cid], threshold
                )

        if self.defense_mode == "rel_cons_agr" and agr:
            threshold = self._threshold([agr.get(cid, 1.0) for cid in selected_ids])
            for cid in selected_ids:
                self.agreement_trust[cid] = self._update_one_score(
                    self.agreement_trust[cid], agr.get(cid, 1.0), threshold
                )

        self._record_histories()

    def _record_histories(self) -> None:
        for cid in self.centroids:
            self.reputation_history[cid].append(self.reputation[cid])
            self.consistency_history[cid].append(self.consistency_trust[cid])
            self.agreement_history[cid].append(self.agreement_trust[cid])

    def get_effective_score(self, client_id: int, top_k: int = 3) -> float:
        return self._effective_trust(client_id, top_k=top_k)

    def get_trust_summary(self, malicious_ids: List[int]) -> Dict:
        malicious = set(malicious_ids)
        honest = [cid for cid in self.reputation if cid not in malicious]

        def avg(values: Dict[int, float], ids: List[int]) -> float:
            present = [values[cid] for cid in ids if cid in values]
            return float(np.mean(present)) if present else 0.0

        return {
            "defense_mode": self.defense_mode,
            "total_queries": self.query_count,
            "malicious_reputation": {cid: self.reputation[cid] for cid in malicious_ids if cid in self.reputation},
            "honest_avg_reputation": avg(self.reputation, honest),
            "malicious_consistency": {
                cid: self.consistency_trust[cid] for cid in malicious_ids if cid in self.consistency_trust
            },
            "honest_avg_consistency": avg(self.consistency_trust, honest),
            "malicious_agreement": {
                cid: self.agreement_trust[cid] for cid in malicious_ids if cid in self.agreement_trust
            },
            "honest_avg_agreement": avg(self.agreement_trust, honest),
        }
