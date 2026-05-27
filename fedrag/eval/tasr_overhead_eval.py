"""
Runtime and memory overhead benchmark for TASR.

The benchmark measures TASR after query/profile/evidence embeddings are already
available. Embedding computation and retrieval are setup costs here because they
are part of the base RAG pipeline rather than TASR's post-routing feedback layer.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from json import JSONDecodeError
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import DownloadMode, load_dataset
from datasets.exceptions import DatasetGenerationError
from tqdm import tqdm


from fedrag.rag.retriever import (  # noqa: E402
    compute_kmeans_profile,
    compute_mean_profile,
    compute_similarity_with_profile,
    embed_queries,
    embed_texts,
    get_embedding_dim,
    load_embedder,
)


DOMAIN_LIST = [
    "electronics",
    "datascience",
    "gaming",
    "academia",
    "chemistry",
    "history",
    "economics",
    "law",
    "cs",
    "biology",
    "mathematica",
    "physics",
    "softwareengineering",
    "security",
    "travel",
    "movies",
    "webapps",
    "gis",
    "android",
    "photo",
]

DOMAIN_TEXT_CACHE: Dict[str, List[str]] = {}
DATASET_CACHE_DIR: Optional[str] = None
DATASET_FALLBACK_CACHE_DIR: Optional[str] = None
DATASET_REDOWNLOAD_CACHE_DIR: Optional[str] = None
DATASET_CACHE_ERRORS = (JSONDecodeError, DatasetGenerationError, OSError)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> List[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def seed32(value: int) -> int:
    return int(value) % (2**32 - 1)


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


def ms_since(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _load_stackexchange_dataset(
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


def load_domain_texts(domain: str) -> List[str]:
    if domain not in DOMAIN_TEXT_CACHE:
        try:
            ds = _load_stackexchange_dataset(domain, DATASET_CACHE_DIR)
        except DATASET_CACHE_ERRORS as exc:
            if not DATASET_FALLBACK_CACHE_DIR:
                raise
            os.makedirs(DATASET_FALLBACK_CACHE_DIR, exist_ok=True)
            print(
                f"[datasets] Cache/load failure for StackExchange/{domain}: "
                f"{type(exc).__name__}. Retrying with {DATASET_FALLBACK_CACHE_DIR}"
            )
            try:
                ds = _load_stackexchange_dataset(domain, DATASET_FALLBACK_CACHE_DIR)
            except DATASET_CACHE_ERRORS as retry_exc:
                if not DATASET_REDOWNLOAD_CACHE_DIR:
                    raise
                redownload_dir = os.path.join(DATASET_REDOWNLOAD_CACHE_DIR, domain)
                os.makedirs(redownload_dir, exist_ok=True)
                print(
                    f"[datasets] Fallback cache also failed for StackExchange/{domain}: "
                    f"{type(retry_exc).__name__}. Forcing fresh download into {redownload_dir}"
                )
                try:
                    ds = _load_stackexchange_dataset(
                        domain,
                        redownload_dir,
                        force_redownload=True,
                    )
                except DATASET_CACHE_ERRORS as final_exc:
                    raise RuntimeError(
                        "Failed to load StackExchange dataset after default cache, "
                        "fallback cache, and forced redownload attempts."
                    ) from final_exc
        data = ds["train"] if "train" in ds else ds
        DOMAIN_TEXT_CACHE[domain] = list(data["title_body"])
    return DOMAIN_TEXT_CACHE[domain]


def take_window(domain: str, n_docs: int, offset: int = 0) -> List[str]:
    texts = load_domain_texts(domain)
    if not texts or n_docs <= 0:
        return []
    start = int(offset) % len(texts)
    end = start + int(n_docs)
    if end <= len(texts):
        return texts[start:end]
    return texts[start:] + texts[: end - len(texts)]


def multidomain_cache_dir(
    cache_dir: str,
    pid: int,
    domains: List[str],
    domains_per_client: int,
) -> Tuple[str, List[str]]:
    rng = np.random.RandomState(seed32(pid))
    selected = rng.choice(
        domains,
        size=min(domains_per_client, len(domains)),
        replace=False,
    ).tolist()
    domain_str = "+".join(sorted(selected))
    return os.path.join(cache_dir, f"multi-{domain_str}-p{pid}"), sorted(selected)


def profile_from_embeddings(
    embeddings: np.ndarray,
    method: str,
    n_clusters: int,
    seed: int,
    sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if sample_size and sample_size > 0 and embeddings.shape[0] > sample_size:
        rng = np.random.RandomState(seed32(seed))
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[idx]

    if method == "kmeans":
        profile = compute_kmeans_profile(embeddings, n_clusters=n_clusters, seed=seed)
        route_centroid = compute_mean_profile(profile)
    else:
        route_centroid = compute_mean_profile(embeddings)
        profile = route_centroid.reshape(1, -1)

    return normalize_rows(route_centroid), normalize_rows(profile)


def build_clients(args: argparse.Namespace, domains: List[str]) -> Dict[int, Dict]:
    clients = {}
    for pid in tqdm(range(args.num_clients), desc="Preparing clients"):
        cache_path, selected_domains = multidomain_cache_dir(
            args.cache_dir,
            pid,
            domains,
            args.domains_per_client,
        )
        embeddings_path = os.path.join(cache_path, "embeddings.npy")

        if args.use_kb_cache and os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            source = "cache"
        else:
            docs = []
            for domain in selected_domains:
                docs.extend(take_window(domain, args.honest_docs_per_domain, offset=0))
            if not docs:
                raise RuntimeError(f"No documents loaded for client {pid}")
            embeddings = embed_texts(docs)
            source = "dataset"

        embeddings = normalize_rows(embeddings)
        route_centroid, profile = profile_from_embeddings(
            embeddings,
            args.profile_method,
            args.profile_n_clusters,
            seed=pid,
            sample_size=args.profile_sample_size,
        )

        if embeddings.shape[0] > args.feedback_docs_per_client:
            rng = np.random.RandomState(seed32(pid + 50_000))
            idx = rng.choice(
                embeddings.shape[0],
                size=args.feedback_docs_per_client,
                replace=False,
            )
            feedback_embeddings = embeddings[idx]
        else:
            feedback_embeddings = embeddings

        clients[pid] = {
            "pid": pid,
            "domains": selected_domains,
            "route_centroid": route_centroid,
            "profile": normalize_rows(profile),
            "method": args.profile_method,
            "doc_embeddings": normalize_rows(feedback_embeddings),
            "source": source,
        }
    return clients


def make_queries(target_domain: str, query_offset: int, num_queries: int) -> List[str]:
    texts = load_domain_texts(target_domain)
    if not texts:
        return []
    start = min(max(int(query_offset), 0), len(texts))
    pool = texts[start:] or texts
    if len(pool) >= num_queries:
        return pool[:num_queries]
    reps = (num_queries + len(pool) - 1) // len(pool)
    return (pool * reps)[:num_queries]


def precompute_evidence(
    clients: Dict[int, Dict],
    query_embeddings: np.ndarray,
    max_feedback_docs: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    evidence = {}
    for qi, qemb in enumerate(tqdm(query_embeddings, desc="Precomputing evidence"), start=1):
        for cid, client in clients.items():
            docs = client["doc_embeddings"]
            sims = docs @ qemb
            k = min(max_feedback_docs, len(sims))
            if k <= 0:
                evidence[(qi, cid)] = np.zeros((0, query_embeddings.shape[1]), dtype=np.float32)
                continue
            if len(sims) > k:
                idx = np.argpartition(sims, -k)[-k:]
            else:
                idx = np.arange(len(sims))
            evidence[(qi, cid)] = docs[idx]
    return evidence


def compute_raw_scores(
    clients: Dict[int, Dict],
    query_emb: np.ndarray,
    top_k: int,
) -> Tuple[Dict[int, float], List[int]]:
    raw_scores = {}
    for cid, client in clients.items():
        raw_scores[cid] = compute_similarity_with_profile(
            query_emb,
            client["profile"],
            client["method"],
        )
    ranked = sorted(raw_scores.keys(), key=lambda cid: raw_scores[cid], reverse=True)
    return raw_scores, ranked[:top_k]


class TASRState:
    def __init__(self, client_ids: Iterable[int], args: argparse.Namespace):
        self.client_ids = list(client_ids)
        self.reputation = {cid: 1.0 for cid in self.client_ids}
        self.consistency = {cid: 1.0 for cid in self.client_ids}
        self.agreement = {cid: 1.0 for cid in self.client_ids}
        self.feedback_count = {cid: 0 for cid in self.client_ids}
        self.query_count = 0

        self.decay_factor = args.decay_factor
        self.recovery_factor = args.recovery_factor
        self.min_reputation = args.min_reputation
        self.warmup_queries = args.warmup_queries
        self.cold_start_s0 = args.cold_start_s0
        self.cold_start_tau = args.cold_start_tau
        self.alpha_r = args.alpha_r
        self.alpha_c = args.alpha_c
        self.alpha_a = args.alpha_a
        self.delta_c = args.delta_c
        self.delta_a = args.delta_a
        self.cons_winner_weight = args.cons_winner_weight
        self.threshold_mode = args.threshold_mode
        self.fixed_threshold = args.fixed_threshold

    def cold_start_factor(self, cid: int) -> float:
        n = self.feedback_count.get(cid, 0)
        return self.cold_start_s0 + (1.0 - self.cold_start_s0) * (
            1.0 - np.exp(-n / self.cold_start_tau)
        )

    @staticmethod
    def soft_gate(x: float, alpha: float, delta: float) -> float:
        return delta + (1.0 - delta) * (x ** alpha)

    def effective_trust(self, cid: int, top_k: int) -> float:
        score = self.cold_start_factor(cid)
        score *= self.reputation[cid] ** self.alpha_r
        score *= self.soft_gate(self.consistency[cid], self.alpha_c, self.delta_c)
        if top_k >= 2:
            score *= self.soft_gate(self.agreement[cid], self.alpha_a, self.delta_a)
        return score

    def update_one(self, current: float, feedback: float, threshold: float) -> float:
        if feedback < threshold:
            return max(current * self.decay_factor, self.min_reputation)
        return min(current * self.recovery_factor, 1.0)


def score_reweighting(
    raw_scores: Dict[int, float],
    state: TASRState,
    top_k: int,
) -> Tuple[Dict[int, float], List[int]]:
    weighted = {}
    for cid, raw in raw_scores.items():
        weighted[cid] = raw * state.effective_trust(cid, top_k)
    ranked = sorted(weighted.keys(), key=lambda cid: weighted[cid], reverse=True)
    return weighted, ranked[:top_k]


def relevance_consistency_scoring(
    clients: Dict[int, Dict],
    query_emb: np.ndarray,
    selected_ids: List[int],
    returned_docs: Dict[int, np.ndarray],
    state: TASRState,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    rel_feedbacks = {}
    cons_feedbacks = {}

    for cid in selected_ids:
        docs = returned_docs[cid]
        if len(docs) == 0:
            rel_feedbacks[cid] = 0.0
            cons_feedbacks[cid] = 0.0
            continue

        rel_feedbacks[cid] = float(np.mean(docs @ query_emb))
        centroids = clients[cid]["profile"]
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        query_sims = centroids @ query_emb
        winning_idx = int(np.argmax(query_sims))

        if centroids.shape[0] == 1:
            cons_feedbacks[cid] = float(np.mean(docs @ centroids[winning_idx]))
            continue

        w = state.cons_winner_weight
        other_weight = (1.0 - w) / (centroids.shape[0] - 1)
        all_sims = centroids @ docs.T
        weighted_sims = w * all_sims[winning_idx]
        for k in range(centroids.shape[0]):
            if k != winning_idx:
                weighted_sims = weighted_sims + other_weight * all_sims[k]
        cons_feedbacks[cid] = float(np.mean(weighted_sims))

    return rel_feedbacks, cons_feedbacks


def agreement_scoring(
    query_emb: np.ndarray,
    selected_ids: List[int],
    returned_docs: Dict[int, np.ndarray],
    rel_feedbacks: Dict[int, float],
    state: TASRState,
) -> Dict[int, float]:
    if state.threshold_mode == "dynamic":
        rel_vals = [rel_feedbacks[cid] for cid in selected_ids]
        rel_threshold = float(np.median(rel_vals)) if rel_vals else 0.5
    else:
        rel_threshold = state.fixed_threshold

    relevant_ids = [
        cid for cid in selected_ids
        if rel_feedbacks.get(cid, 0.0) >= rel_threshold
    ]
    if len(relevant_ids) < 2:
        return {cid: 1.0 for cid in selected_ids}

    doc_centroids = {}
    for cid in relevant_ids:
        docs = returned_docs[cid]
        if len(docs) == 0:
            doc_centroids[cid] = None
            continue
        centroid = docs.mean(axis=0)
        doc_centroids[cid] = centroid / (np.linalg.norm(centroid) + 1e-8)

    agreement = {}
    for cid in selected_ids:
        if cid not in relevant_ids or doc_centroids.get(cid) is None:
            agreement[cid] = 1.0
            continue
        weighted_sum = 0.0
        weight_total = 0.0
        for other_cid in relevant_ids:
            if other_cid == cid or doc_centroids.get(other_cid) is None:
                continue
            w = state.reputation[other_cid]
            weighted_sum += w * float(np.dot(doc_centroids[cid], doc_centroids[other_cid]))
            weight_total += w
        agreement[cid] = weighted_sum / weight_total if weight_total > 1e-12 else 1.0
    return agreement


def trust_update(
    selected_ids: List[int],
    rel_feedbacks: Dict[int, float],
    cons_feedbacks: Dict[int, float],
    agr_feedbacks: Dict[int, float],
    state: TASRState,
) -> None:
    state.query_count += 1
    for cid in selected_ids:
        state.feedback_count[cid] = state.feedback_count.get(cid, 0) + 1

    if state.query_count <= state.warmup_queries:
        return

    if state.threshold_mode == "dynamic":
        rel_threshold = float(np.median([rel_feedbacks[cid] for cid in selected_ids]))
        cons_threshold = float(np.median([cons_feedbacks[cid] for cid in selected_ids]))
        agr_threshold = float(np.median([agr_feedbacks.get(cid, 1.0) for cid in selected_ids]))
    else:
        rel_threshold = state.fixed_threshold
        cons_threshold = state.fixed_threshold
        agr_threshold = state.fixed_threshold

    for cid in selected_ids:
        state.reputation[cid] = state.update_one(
            state.reputation[cid],
            rel_feedbacks[cid],
            rel_threshold,
        )
        state.consistency[cid] = state.update_one(
            state.consistency[cid],
            cons_feedbacks[cid],
            cons_threshold,
        )
        state.agreement[cid] = state.update_one(
            state.agreement[cid],
            agr_feedbacks.get(cid, 1.0),
            agr_threshold,
        )


def run_one_pass(
    clients: Dict[int, Dict],
    query_embeddings: np.ndarray,
    evidence: Dict[Tuple[int, int], np.ndarray],
    route_top_k: int,
    feedback_docs: int,
    args: argparse.Namespace,
    record: bool,
) -> List[Dict]:
    state = TASRState(clients.keys(), args)
    rows = []

    for qi, qemb in enumerate(query_embeddings, start=1):
        start = time.perf_counter_ns()
        raw_scores, _ = compute_raw_scores(clients, qemb, route_top_k)
        base_ms = ms_since(start)

        start = time.perf_counter_ns()
        _, selected = score_reweighting(raw_scores, state, route_top_k)
        reweight_ms = ms_since(start)

        selected_docs = {
            cid: evidence[(qi, cid)][:feedback_docs]
            for cid in selected
        }

        start = time.perf_counter_ns()
        rel_feedbacks, cons_feedbacks = relevance_consistency_scoring(
            clients,
            qemb,
            selected,
            selected_docs,
            state,
        )
        rel_cons_ms = ms_since(start)

        start = time.perf_counter_ns()
        agr_feedbacks = agreement_scoring(
            qemb,
            selected,
            selected_docs,
            rel_feedbacks,
            state,
        )
        agreement_ms = ms_since(start)

        start = time.perf_counter_ns()
        trust_update(
            selected,
            rel_feedbacks,
            cons_feedbacks,
            agr_feedbacks,
            state,
        )
        update_ms = ms_since(start)

        overhead_ms = reweight_ms + rel_cons_ms + agreement_ms + update_ms
        if record:
            rows.append({
                "route_top_k": route_top_k,
                "feedback_docs": feedback_docs,
                "query_idx": qi,
                "base_routing_ms": base_ms,
                "score_reweighting_ms": reweight_ms,
                "relevance_consistency_ms": rel_cons_ms,
                "agreement_scoring_ms": agreement_ms,
                "trust_update_ms": update_ms,
                "full_tasr_overhead_ms": overhead_ms,
                "full_tasr_total_ms": base_ms + overhead_ms,
                "selected": " ".join(str(cid) for cid in selected),
            })

    return rows


def summarize_details(rows: List[Dict]) -> List[Dict]:
    components = [
        ("base_routing", "base_routing_ms"),
        ("score_reweighting", "score_reweighting_ms"),
        ("relevance_consistency_scoring", "relevance_consistency_ms"),
        ("agreement_scoring", "agreement_scoring_ms"),
        ("trust_update", "trust_update_ms"),
        ("full_tasr_overhead", "full_tasr_overhead_ms"),
        ("full_tasr_total", "full_tasr_total_ms"),
    ]
    grouped: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["route_top_k"], row["feedback_docs"]), []).append(row)

    summary = []
    for (route_top_k, feedback_docs), items in sorted(grouped.items()):
        base_mean = mean(float(row["base_routing_ms"]) for row in items)
        overhead_mean = mean(float(row["full_tasr_overhead_ms"]) for row in items)
        for component, field in components:
            vals = np.array([float(row[field]) for row in items], dtype=np.float64)
            component_mean = float(np.mean(vals))
            component_p95 = float(np.percentile(vals, 95))
            component_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if component == "base_routing":
                incremental = 0.0
            elif component == "full_tasr_total":
                incremental = (overhead_mean / base_mean * 100.0) if base_mean > 0 else 0.0
            else:
                incremental = (component_mean / base_mean * 100.0) if base_mean > 0 else 0.0
            summary.append({
                "route_top_k": route_top_k,
                "feedback_docs": feedback_docs,
                "component": component,
                "n_queries": len(items),
                "mean_ms_per_query": component_mean,
                "p95_ms_per_query": component_p95,
                "std_ms_per_query": component_std,
                "percent_of_base_routing": (component_mean / base_mean * 100.0) if base_mean > 0 else 0.0,
                "incremental_overhead_over_base_percent": incremental,
            })
    return summary


def memory_summary(num_clients: int) -> Dict:
    bytes_per_client_float32 = 4 * 4
    return {
        "persistent_state_model": "3 trust scalars + 1 feedback/cold-start count per client",
        "persistent_scalars_per_client": 4,
        "algorithmic_complexity": "O(|C|)",
        "bytes_per_client_float32_or_int32": bytes_per_client_float32,
        "total_bytes_float32_or_int32": num_clients * bytes_per_client_float32,
        "total_kb_float32_or_int32": num_clients * bytes_per_client_float32 / 1024.0,
        "note": (
            "Evaluation history and precomputed evidence arrays are benchmark/logging "
            "artifacts, not required TASR deployment state."
        ),
    }


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_overhead(args: argparse.Namespace) -> Dict:
    global DATASET_CACHE_DIR, DATASET_FALLBACK_CACHE_DIR, DATASET_REDOWNLOAD_CACHE_DIR

    DATASET_CACHE_DIR = args.dataset_cache_dir or None
    DATASET_FALLBACK_CACHE_DIR = (
        args.dataset_fallback_cache_dir
        or os.path.join(args.output_dir, "hf_datasets_cache_tasr_overhead")
    )
    DATASET_REDOWNLOAD_CACHE_DIR = os.path.join(
        args.output_dir,
        f"hf_datasets_redownload_tasr_overhead_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    domains = parse_str_list(args.domains)
    route_top_k_list = parse_int_list(args.route_top_k_list)
    feedback_docs_list = parse_int_list(args.feedback_docs_list)
    max_feedback_docs = max(feedback_docs_list)

    load_embedder(args.emb_model, model_type=args.emb_model_type)
    embedding_dim = get_embedding_dim()
    print(f"Embedding dimension: {embedding_dim}")

    clients = build_clients(args, domains)
    queries = make_queries(args.target_domain, args.query_offset, args.num_queries)
    print(f"Encoding {len(queries)} queries for target={args.target_domain}")
    query_embeddings = normalize_rows(embed_queries(queries))
    if query_embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Query embedding dim mismatch: {query_embeddings.shape[1]} vs {embedding_dim}"
        )

    evidence = precompute_evidence(clients, query_embeddings, max_feedback_docs)

    all_details = []
    for route_top_k in route_top_k_list:
        for feedback_docs in feedback_docs_list:
            print(f"Benchmarking route_top_k={route_top_k}, feedback_docs={feedback_docs}")
            if args.warmup_iterations > 0:
                warm_queries = query_embeddings[: min(args.warmup_iterations, len(query_embeddings))]
                run_one_pass(
                    clients,
                    warm_queries,
                    {
                        (qi, cid): evidence[(qi, cid)]
                        for qi in range(1, len(warm_queries) + 1)
                        for cid in clients
                    },
                    route_top_k,
                    feedback_docs,
                    args,
                    record=False,
                )
            rows = run_one_pass(
                clients,
                query_embeddings,
                evidence,
                route_top_k,
                feedback_docs,
                args,
                record=True,
            )
            all_details.extend(rows)

    summary = summarize_details(all_details)
    mem = memory_summary(len(clients))
    config = {
        "router": "embedding_based_routing",
        "topology": "multi-domain K-Means profiles",
        "num_clients": args.num_clients,
        "domains_per_client": args.domains_per_client,
        "num_queries": args.num_queries,
        "target_domain": args.target_domain,
        "route_top_k_list": route_top_k_list,
        "feedback_docs_list": feedback_docs_list,
        "profile_method": args.profile_method,
        "profile_n_clusters": args.profile_n_clusters,
        "profile_sample_size": args.profile_sample_size,
        "feedback_docs_per_client": args.feedback_docs_per_client,
        "embedding_dim": embedding_dim,
        "emb_model": args.emb_model,
        "emb_model_type": args.emb_model_type,
        "timing_scope": (
            "Embeddings and returned evidence arrays are precomputed before timing; "
            "timed measurements cover base routing and TASR layer computations only."
        ),
        "dataset_cache_dir": DATASET_CACHE_DIR,
        "dataset_fallback_cache_dir": DATASET_FALLBACK_CACHE_DIR,
        "dataset_redownload_cache_dir": DATASET_REDOWNLOAD_CACHE_DIR,
    }
    return {
        "config": config,
        "details": all_details,
        "summary": summary,
        "memory": mem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TASR runtime and memory overhead benchmark")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--domains", default=",".join(DOMAIN_LIST))
    parser.add_argument("--domains-per-client", type=int, default=3)
    parser.add_argument("--honest-docs-per-domain", type=int, default=10000)
    parser.add_argument("--target-domain", default="physics")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--query-offset", type=int, default=30000)

    parser.add_argument("--route-top-k-list", default="3")
    parser.add_argument("--feedback-docs-list", default="5")
    parser.add_argument("--feedback-docs-per-client", type=int, default=500)

    parser.add_argument("--profile-method", choices=["mean", "kmeans"], default="kmeans")
    parser.add_argument("--profile-n-clusters", type=int, default=5)
    parser.add_argument("--profile-sample-size", type=int, default=0)

    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--cold-start-s0", type=float, default=0.7)
    parser.add_argument("--cold-start-tau", type=float, default=30.0)
    parser.add_argument("--decay-factor", type=float, default=0.9)
    parser.add_argument("--recovery-factor", type=float, default=1.02)
    parser.add_argument("--min-reputation", type=float, default=0.01)
    parser.add_argument("--threshold-mode", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-c", type=float, default=1.0)
    parser.add_argument("--alpha-a", type=float, default=0.5)
    parser.add_argument("--delta-c", type=float, default=0.3)
    parser.add_argument("--delta-a", type=float, default=0.5)
    parser.add_argument("--cons-winner-weight", type=float, default=0.6)

    parser.add_argument("--emb-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--emb-model-type", default="sentence-transformer")
    parser.add_argument("--cache-dir", default="kb-cache")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-fallback-cache-dir", default=None)
    parser.add_argument(
        "--no-kb-cache",
        action="store_false",
        dest="use_kb_cache",
        help="Ignore cached profile embeddings and rebuild from datasets.",
    )
    parser.set_defaults(use_kb_cache=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_overhead(args)

    details_path = os.path.join(args.output_dir, f"tasr_overhead_details_{timestamp}.csv")
    summary_path = os.path.join(args.output_dir, f"tasr_overhead_summary_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"tasr_overhead_summary_{timestamp}.json")

    write_csv(details_path, result["details"])
    write_csv(summary_path, result["summary"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": result["config"],
                "summary": result["summary"],
                "memory": result["memory"],
                "files": {
                    "details_csv": details_path,
                    "summary_csv": summary_path,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTASR overhead benchmark completed.")
    print(f"  Details: {details_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Summary JSON: {json_path}")
    print(
        "  Persistent state: "
        f"{result['memory']['total_bytes_float32_or_int32']} bytes "
        f"({result['memory']['total_kb_float32_or_int32']:.4f} KB) for "
        f"{args.num_clients} clients"
    )


if __name__ == "__main__":
    main()
