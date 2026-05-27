"""
TASR transfer evaluation on the RAGRoute-style neural router.

This script treats each StackExchange domain as one federated source. The base
router is the trained NNRouter MLP scorer. TASR is applied as a post-routing
feedback layer by multiplying the neural router score by the current trust
weight.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import DownloadMode, load_dataset
from datasets.exceptions import DatasetGenerationError
from tqdm import tqdm


from fedrag.rag.nn_router import NNRouter  # noqa: E402
from fedrag.rag.retriever import (  # noqa: E402
    compute_mean_profile,
    embed_queries,
    embed_texts,
    load_embedder,
)
from fedrag.rag.trust_defense import TrustAwareRouter  # noqa: E402


DOMAIN_TEXT_CACHE: Dict[str, List[str]] = {}
DATASET_CACHE_DIR: Optional[str] = None
DATASET_FALLBACK_CACHE_DIR: Optional[str] = None
DATASET_REDOWNLOAD_CACHE_DIR: Optional[str] = None
DATASET_CACHE_ERRORS = (JSONDecodeError, DatasetGenerationError, OSError)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


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


def make_queries(domain: str, query_offset: int, num_queries: int) -> List[str]:
    texts = load_domain_texts(domain)
    if not texts:
        return []
    start = min(max(int(query_offset), 0), len(texts))
    pool = texts[start:] or texts
    if len(pool) >= num_queries:
        return pool[:num_queries]
    reps = (num_queries + len(pool) - 1) // len(pool)
    return (pool * reps)[:num_queries]


def sample_domain(domain: str, n_docs: int, offset: int, seed: int) -> List[str]:
    texts = load_domain_texts(domain)
    if not texts or n_docs <= 0:
        return []
    start = min(max(int(offset), 0), len(texts))
    pool = texts[start:] or texts
    rng = np.random.RandomState(seed32(seed))
    replace = len(pool) < n_docs
    idx = rng.choice(len(pool), size=n_docs, replace=replace)
    return [pool[int(i)] for i in idx]


def load_router_centroids(path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "centroids" not in data or "domain_list" not in data:
        raise ValueError(f"Centroids file has unexpected format: {path}")
    centroids = {
        str(domain).lower(): normalize_rows(vec)
        for domain, vec in data["centroids"].items()
    }
    domain_list = [str(d).lower() for d in data["domain_list"]]
    return centroids, domain_list


def find_cached_embeddings(cache_dir: str, domain: str) -> Optional[Path]:
    root = Path(cache_dir)
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"{domain}-p*/{domain}/embeddings.npy"))
    if candidates:
        return candidates[0]
    return None


def load_feedback_embeddings(
    domain: str,
    args: argparse.Namespace,
    expected_dim: int,
) -> np.ndarray:
    if args.use_kb_cache:
        cached = find_cached_embeddings(args.cache_dir, domain)
        if cached is not None:
            emb = np.load(cached)
            if emb.ndim == 2 and emb.shape[1] == expected_dim:
                return normalize_rows(emb)
            print(
                f"[cache] Ignoring {cached} for domain={domain}: "
                f"dim={emb.shape[1] if emb.ndim == 2 else 'invalid'}, expected={expected_dim}"
            )

    docs = make_queries(domain, 0, max(args.feedback_docs_per_source, args.fallback_docs_per_source))
    if not docs:
        raise RuntimeError(f"No feedback documents available for domain={domain}")
    return normalize_rows(embed_texts(docs))


def sample_feedback_pool(
    embeddings: np.ndarray,
    max_docs: int,
    seed: int,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] <= max_docs:
        return normalize_rows(embeddings)
    rng = np.random.RandomState(seed32(seed))
    idx = rng.choice(embeddings.shape[0], size=max_docs, replace=False)
    return normalize_rows(embeddings[idx])


def build_forged_centroid(
    target_domain: str,
    proxy_size: int,
    proxy_offset: int,
    seed: int,
) -> np.ndarray:
    docs = sample_domain(target_domain, proxy_size, proxy_offset, seed)
    if not docs:
        raise RuntimeError(f"No proxy documents for target={target_domain}")
    emb = embed_texts(docs)
    return normalize_rows(compute_mean_profile(np.asarray(emb, dtype=np.float32)))


def select_malicious_domains(
    source_domains: List[str],
    target_domain: str,
    num_malicious: int,
    seed: int,
) -> List[str]:
    candidates = [d for d in source_domains if d != target_domain]
    if len(candidates) < num_malicious:
        raise ValueError(
            f"Need {num_malicious} malicious sources, but only {len(candidates)} candidates remain"
        )
    rng = np.random.RandomState(seed32(seed))
    chosen = rng.choice(candidates, size=num_malicious, replace=False)
    return sorted(str(x) for x in chosen)


class RAGRouteTrustRouter(TrustAwareRouter):
    """TASR wrapper whose base scores come from NNRouter probabilities."""

    def __init__(
        self,
        nn_router: NNRouter,
        source_domains: List[str],
        domain_to_id: Dict[str, int],
        centroids_override: Dict[str, np.ndarray],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nn_router = nn_router
        self.source_domains = source_domains
        self.domain_to_id = domain_to_id
        self.id_to_domain = {v: k for k, v in domain_to_id.items()}
        self.centroids_override = {
            domain: normalize_rows(vec)
            for domain, vec in centroids_override.items()
        }
        self.last_weighted_scores: Dict[int, float] = {}
        self.last_raw_scores: Dict[int, float] = {}

    def register_source(
        self,
        domain: str,
        centroid: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> None:
        cid = self.domain_to_id[domain]
        self.register_client(
            cid,
            normalize_rows(centroid),
            normalize_rows(doc_embeddings),
            profile_centroids=normalize_rows(centroid).reshape(1, -1),
        )

    def route(self, query_emb: np.ndarray, top_k: int = 3):
        query_emb = normalize_rows(query_emb)
        probs = self.nn_router.predict_all_domains(query_emb, self.centroids_override)

        raw_scores = {}
        weighted_scores = {}
        for domain in self.source_domains:
            cid = self.domain_to_id[domain]
            raw = max(float(probs.get(domain, 0.0)), 0.0)
            raw_scores[cid] = raw

            weight = raw * self._cold_start_factor(cid)
            if self.defense_mode != "none":
                weight *= self.reputation[cid] ** self.alpha_r
            if self.defense_mode in ("rel_cons", "rel_cons_agr"):
                weight *= self._soft_gate(
                    self.consistency_trust[cid],
                    self.alpha_c,
                    self.delta_c,
                )
            if self.defense_mode == "rel_cons_agr" and top_k >= 2:
                weight *= self._soft_gate(
                    self.agreement_trust[cid],
                    self.alpha_a,
                    self.delta_a,
                )
            weighted_scores[cid] = weight

        sorted_clients = sorted(
            weighted_scores.keys(),
            key=lambda cid: weighted_scores[cid],
            reverse=True,
        )
        selected = sorted_clients[:top_k]

        if (
            self.explore_interval > 0
            and self.query_count > 0
            and self.query_count % self.explore_interval == 0
        ):
            remaining = [cid for cid in sorted_clients[top_k:] if cid not in selected]
            if remaining:
                selected = selected + remaining[: self.explore_extra]

        self.last_raw_scores = raw_scores
        self.last_weighted_scores = weighted_scores
        return selected, raw_scores


def build_router(
    nn_router: NNRouter,
    source_domains: List[str],
    domain_to_id: Dict[str, int],
    centroids_override: Dict[str, np.ndarray],
    feedback_embeddings: Dict[str, np.ndarray],
    mode: str,
    args: argparse.Namespace,
) -> RAGRouteTrustRouter:
    router = RAGRouteTrustRouter(
        nn_router,
        source_domains,
        domain_to_id,
        centroids_override,
        decay_factor=args.decay_factor,
        recovery_factor=args.recovery_factor,
        min_reputation=args.min_reputation,
        warmup_queries=args.warmup_queries,
        cold_start_s0=args.cold_start_s0,
        cold_start_tau=args.cold_start_tau,
        docs_for_feedback=args.feedback_docs,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        alpha_r=args.alpha_r,
        alpha_c=args.alpha_c,
        alpha_a=args.alpha_a,
        delta_c=args.delta_c,
        delta_a=args.delta_a,
        cons_winner_weight=args.cons_winner_weight,
        explore_interval=args.explore_interval,
        explore_extra=args.explore_extra,
        defense_mode=mode,
    )
    for domain in source_domains:
        router.register_source(
            domain,
            centroids_override[domain],
            feedback_embeddings[domain],
        )
    return router


def is_hijack(selected: List[int], malicious_ids: Iterable[int], k: int) -> bool:
    mal = set(malicious_ids)
    return any(cid in mal for cid in selected[:k])


def is_correct(selected: List[int], target_ids: Iterable[int], k: int) -> bool:
    targets = set(target_ids)
    return any(cid in targets for cid in selected[:k])


def avg_dict_values(values: Dict[int, float], ids: Iterable[int]) -> float:
    present = [float(values[i]) for i in ids if i in values]
    return mean(present) if present else 0.0


def avg_effective(router: RAGRouteTrustRouter, ids: Iterable[int], top_k: int) -> float:
    vals = [router.get_effective_score(i, top_k=top_k) for i in ids if i in router.reputation]
    return mean(vals) if vals else 0.0


def malicious_rank(router: RAGRouteTrustRouter, malicious_ids: Iterable[int]) -> int:
    if not router.last_weighted_scores:
        return 0
    ranked = sorted(
        router.last_weighted_scores.keys(),
        key=lambda cid: router.last_weighted_scores[cid],
        reverse=True,
    )
    rank_map = {cid: idx + 1 for idx, cid in enumerate(ranked)}
    default_rank = len(ranked) + 1
    return min(rank_map.get(cid, default_rank) for cid in malicious_ids)


def run_stream(
    nn_router: NNRouter,
    source_domains: List[str],
    domain_to_id: Dict[str, int],
    centroids_override: Dict[str, np.ndarray],
    feedback_embeddings: Dict[str, np.ndarray],
    query_embeddings: np.ndarray,
    target_domain: str,
    malicious_domains: List[str],
    seed: int,
    mode_specs: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    routers = {
        label: build_router(
            nn_router,
            source_domains,
            domain_to_id,
            centroids_override,
            feedback_embeddings,
            mode,
            args,
        )
        for label, mode in mode_specs
    }

    target_ids = [domain_to_id[target_domain]]
    malicious_ids = [domain_to_id[d] for d in malicious_domains]
    honest_ids = [domain_to_id[d] for d in source_domains if d not in malicious_domains]

    detail_rows = []
    window_rows = []
    aggregate_rows = []
    running = {
        label: {"h1": 0, "h3": 0, "a1": 0, "a3": 0, "rank_sum": 0.0, "window": []}
        for label, _ in mode_specs
    }

    for qi, qemb in enumerate(query_embeddings, start=1):
        for label, _ in mode_specs:
            router = routers[label]
            selected, _ = router.route(qemb, top_k=args.route_top_k)
            k3 = min(3, args.route_top_k)
            mrank = malicious_rank(router, malicious_ids)
            row = {
                "experiment": "ragroute_online",
                "router": "ragroute_nn_tasr",
                "target_domain": target_domain,
                "num_malicious": len(malicious_domains),
                "malicious_domains": " ".join(malicious_domains),
                "seed": seed,
                "mode": label,
                "warmup_queries": args.warmup_queries,
                "cold_start_s0": args.cold_start_s0,
                "query_idx": qi,
                "window_idx": (qi - 1) // args.window_size + 1,
                "selected_ids": " ".join(str(x) for x in selected),
                "selected_domains": " ".join(router.id_to_domain[x] for x in selected),
                "top1_domain": router.id_to_domain[selected[0]] if selected else "",
                "hijack@1": int(is_hijack(selected, malicious_ids, 1)),
                "hijack@3": int(is_hijack(selected, malicious_ids, k3)),
                "acc@1": int(is_correct(selected, target_ids, 1)),
                "acc@3": int(is_correct(selected, target_ids, k3)),
                "malicious_rank": mrank,
            }
            detail_rows.append(row)
            running[label]["h1"] += row["hijack@1"]
            running[label]["h3"] += row["hijack@3"]
            running[label]["a1"] += row["acc@1"]
            running[label]["a3"] += row["acc@3"]
            running[label]["rank_sum"] += mrank
            running[label]["window"].append(row)

            if label != "No Defense":
                router.update_trust(qemb, selected)

        if qi % args.window_size == 0 or qi == len(query_embeddings):
            for label, _ in mode_specs:
                router = routers[label]
                wrows = running[label]["window"]
                wcount = len(wrows)
                window_rows.append({
                    "experiment": "ragroute_online",
                    "target_domain": target_domain,
                    "num_malicious": len(malicious_domains),
                    "mode": label,
                    "warmup_queries": args.warmup_queries,
                    "cold_start_s0": args.cold_start_s0,
                    "seed": seed,
                    "window_idx": (qi - 1) // args.window_size + 1,
                    "query_start": qi - wcount + 1,
                    "query_end": qi,
                    "window_size": wcount,
                    "window_hr@1": sum(r["hijack@1"] for r in wrows) / max(wcount, 1),
                    "window_hr@3": sum(r["hijack@3"] for r in wrows) / max(wcount, 1),
                    "window_acc@1": sum(r["acc@1"] for r in wrows) / max(wcount, 1),
                    "window_acc@3": sum(r["acc@3"] for r in wrows) / max(wcount, 1),
                    "window_malicious_rank": mean([r["malicious_rank"] for r in wrows]) if wrows else 0.0,
                    "cumulative_hr@1": running[label]["h1"] / qi,
                    "cumulative_hr@3": running[label]["h3"] / qi,
                    "cumulative_acc@1": running[label]["a1"] / qi,
                    "cumulative_acc@3": running[label]["a3"] / qi,
                    "cumulative_malicious_rank": running[label]["rank_sum"] / qi,
                    "mal_reputation": avg_dict_values(router.reputation, malicious_ids),
                    "honest_reputation": avg_dict_values(router.reputation, honest_ids),
                    "target_reputation": avg_dict_values(router.reputation, target_ids),
                    "mal_effective_trust": avg_effective(router, malicious_ids, args.route_top_k),
                    "target_effective_trust": avg_effective(router, target_ids, args.route_top_k),
                })
                running[label]["window"] = []

    for label, _ in mode_specs:
        router = routers[label]
        label_rows = [r for r in detail_rows if r["mode"] == label]
        early_end = min(len(label_rows), args.early_queries)
        late_start = max(0, len(label_rows) - args.late_queries)
        early_rows = label_rows[:early_end]
        late_rows = label_rows[late_start:]
        aggregate_rows.append({
            "experiment": "ragroute_online",
            "target_domain": target_domain,
            "num_malicious": len(malicious_domains),
            "mode": label,
            "warmup_queries": args.warmup_queries,
            "cold_start_s0": args.cold_start_s0,
            "seed": seed,
            "total_queries": len(query_embeddings),
            "final_hr@1": running[label]["h1"] / len(query_embeddings),
            "final_hr@3": running[label]["h3"] / len(query_embeddings),
            "final_acc@1": running[label]["a1"] / len(query_embeddings),
            "final_acc@3": running[label]["a3"] / len(query_embeddings),
            "mean_malicious_rank": running[label]["rank_sum"] / len(query_embeddings),
            "early_hr@1": mean([r["hijack@1"] for r in early_rows]) if early_rows else 0.0,
            "early_hr@3": mean([r["hijack@3"] for r in early_rows]) if early_rows else 0.0,
            "early_acc@1": mean([r["acc@1"] for r in early_rows]) if early_rows else 0.0,
            "early_acc@3": mean([r["acc@3"] for r in early_rows]) if early_rows else 0.0,
            "late_hr@1": mean([r["hijack@1"] for r in late_rows]) if late_rows else 0.0,
            "late_hr@3": mean([r["hijack@3"] for r in late_rows]) if late_rows else 0.0,
            "late_acc@1": mean([r["acc@1"] for r in late_rows]) if late_rows else 0.0,
            "late_acc@3": mean([r["acc@3"] for r in late_rows]) if late_rows else 0.0,
            "late_malicious_rank": mean([r["malicious_rank"] for r in late_rows]) if late_rows else 0.0,
            "final_mal_reputation": avg_dict_values(router.reputation, malicious_ids),
            "final_target_reputation": avg_dict_values(router.reputation, target_ids),
            "final_mal_effective_trust": avg_effective(router, malicious_ids, args.route_top_k),
            "final_target_effective_trust": avg_effective(router, target_ids, args.route_top_k),
        })

    return detail_rows, window_rows, aggregate_rows


def aggregate(rows: List[Dict], group_keys: List[str], metric_keys: List[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out = []
    for key, items in sorted(grouped.items(), key=lambda x: x[0]):
        result = {k: v for k, v in zip(group_keys, key)}
        result["n"] = len(items)
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items]
            result[f"{metric}_mean"] = mean(vals)
            result[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(result)
    return out


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_tasr_ragroute(args: argparse.Namespace) -> Dict:
    global DATASET_CACHE_DIR, DATASET_FALLBACK_CACHE_DIR, DATASET_REDOWNLOAD_CACHE_DIR

    DATASET_CACHE_DIR = args.dataset_cache_dir or None
    DATASET_FALLBACK_CACHE_DIR = (
        args.dataset_fallback_cache_dir
        or os.path.join(args.output_dir, "hf_datasets_cache_ragroute_tasr")
    )
    DATASET_REDOWNLOAD_CACHE_DIR = os.path.join(
        args.output_dir,
        f"hf_datasets_redownload_ragroute_tasr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    targets = parse_str_list(args.target_domains)
    seeds = parse_int_list(args.seeds)
    num_malicious_list = parse_int_list(args.num_malicious_list)
    excluded = set(parse_str_list(args.exclude_source_domains))

    centroids, trained_domains = load_router_centroids(args.nn_centroids_path)
    source_domains = [d for d in trained_domains if d not in excluded]
    for target in targets:
        if target not in source_domains:
            raise ValueError(f"Target domain {target} is not available as a RAGRoute source")

    domain_to_id = {domain: idx for idx, domain in enumerate(trained_domains)}

    print(f"Loading embedder: {args.emb_model}")
    load_embedder(args.emb_model, model_type=args.emb_model_type)

    print(f"Loading NNRouter: {args.nn_model_path}")
    nn_router = NNRouter(
        args.nn_model_path,
        args.nn_centroids_path,
        trained_domains,
        args.emb_model,
        device=args.device or None,
    )

    print("Loading feedback embeddings for sources")
    feedback_embeddings = {}
    expected_dim = next(iter(centroids.values())).shape[0]
    for domain in tqdm(source_domains, desc="Feedback sources"):
        pool = load_feedback_embeddings(domain, args, expected_dim)
        if pool.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch for {domain}: got {pool.shape[1]}, expected {expected_dim}. "
                "Use the embedding model that was used to train NNRouter."
            )
        feedback_embeddings[domain] = sample_feedback_pool(
            pool,
            args.feedback_docs_per_source,
            stable_int(domain),
        )

    print("Encoding query streams")
    query_embeddings_by_target = {}
    for target in targets:
        queries = make_queries(target, args.query_offset, args.num_queries)
        print(f"  target={target}: {len(queries)} queries")
        qemb = normalize_rows(embed_queries(queries))
        if qemb.shape[1] != expected_dim:
            raise ValueError(
                f"Query embedding dim mismatch: got {qemb.shape[1]}, expected {expected_dim}. "
                "Use the embedding model that was used to train NNRouter."
            )
        query_embeddings_by_target[target] = qemb

    mode_specs = [
        ("No Defense", "none"),
        ("Rel", "rel"),
        ("Full TASR", "rel_cons_agr"),
    ]
    if args.include_rel_cons_in_main:
        mode_specs.insert(2, ("Rel+Cons", "rel_cons"))

    all_details = []
    all_windows = []
    all_aggregates = []

    for target in targets:
        for num_malicious in num_malicious_list:
            for seed in seeds:
                malicious_domains = select_malicious_domains(
                    source_domains,
                    target,
                    num_malicious,
                    seed + stable_int(target),
                )
                centroids_override = {d: centroids[d] for d in trained_domains if d in centroids}
                for malicious_domain in malicious_domains:
                    centroids_override[malicious_domain] = build_forged_centroid(
                        target,
                        args.proxy_size,
                        args.proxy_offset,
                        seed + stable_int(f"{target}:{malicious_domain}"),
                    )

                details, windows, aggregates = run_stream(
                    nn_router,
                    source_domains,
                    domain_to_id,
                    centroids_override,
                    feedback_embeddings,
                    query_embeddings_by_target[target],
                    target,
                    malicious_domains,
                    seed,
                    mode_specs,
                    args,
                )
                all_details.extend(details)
                all_windows.extend(windows)
                all_aggregates.extend(aggregates)

    aggregate_metrics = [
        "final_hr@1",
        "final_hr@3",
        "final_acc@1",
        "final_acc@3",
        "mean_malicious_rank",
        "early_hr@1",
        "early_hr@3",
        "early_acc@1",
        "early_acc@3",
        "late_hr@1",
        "late_hr@3",
        "late_acc@1",
        "late_acc@3",
        "late_malicious_rank",
        "final_mal_reputation",
        "final_target_reputation",
        "final_mal_effective_trust",
        "final_target_effective_trust",
    ]
    aggregate_summary = aggregate(
        all_aggregates,
        [
            "experiment",
            "target_domain",
            "num_malicious",
            "mode",
            "warmup_queries",
            "cold_start_s0",
        ],
        aggregate_metrics,
    )
    window_summary = aggregate(
        all_windows,
        [
            "experiment",
            "target_domain",
            "num_malicious",
            "mode",
            "warmup_queries",
            "cold_start_s0",
            "window_idx",
            "query_start",
            "query_end",
        ],
        [
            "window_hr@1",
            "window_hr@3",
            "window_acc@1",
            "window_acc@3",
            "window_malicious_rank",
            "cumulative_hr@1",
            "cumulative_hr@3",
            "cumulative_acc@1",
            "cumulative_acc@3",
            "cumulative_malicious_rank",
            "mal_effective_trust",
            "target_effective_trust",
            "mal_reputation",
            "target_reputation",
        ],
    )

    return {
        "config": {
            "router": "ragroute_nn_tasr",
            "nn_model_path": args.nn_model_path,
            "nn_centroids_path": args.nn_centroids_path,
            "emb_model": args.emb_model,
            "source_domains": source_domains,
            "excluded_source_domains": sorted(excluded),
            "target_domains": targets,
            "num_malicious_list": num_malicious_list,
            "seeds": seeds,
            "num_queries": args.num_queries,
            "window_size": args.window_size,
            "route_top_k": args.route_top_k,
            "proxy_size": args.proxy_size,
            "warmup_queries": args.warmup_queries,
            "cold_start_s0": args.cold_start_s0,
            "decay_factor": args.decay_factor,
            "recovery_factor": args.recovery_factor,
            "dataset_cache_dir": DATASET_CACHE_DIR,
            "dataset_fallback_cache_dir": DATASET_FALLBACK_CACHE_DIR,
            "dataset_redownload_cache_dir": DATASET_REDOWNLOAD_CACHE_DIR,
        },
        "details": all_details,
        "windows": all_windows,
        "aggregates": all_aggregates,
        "aggregate_summary": aggregate_summary,
        "window_summary": window_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TASR transfer evaluation on RAGRoute/NNRouter")

    parser.add_argument("--target-domains", default="gaming,gis,physics")
    parser.add_argument("--num-malicious-list", default="1,3")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--early-queries", type=int, default=100)
    parser.add_argument("--late-queries", type=int, default=100)
    parser.add_argument("--route-top-k", type=int, default=3)

    parser.add_argument("--proxy-size", type=int, default=100)
    parser.add_argument("--proxy-offset", type=int, default=30000)
    parser.add_argument("--query-offset", type=int, default=30000)

    parser.add_argument("--nn-model-path", default="nn_router_model.pt")
    parser.add_argument("--nn-centroids-path", default="nn_router_centroids.pkl")
    parser.add_argument("--emb-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--emb-model-type", default="sentence-transformer")
    parser.add_argument("--device", default=None)

    parser.add_argument("--feedback-docs-per-source", type=int, default=500)
    parser.add_argument("--feedback-docs", type=int, default=5)
    parser.add_argument("--fallback-docs-per-source", type=int, default=1000)

    parser.add_argument("--warmup-queries", type=int, default=50)
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
    parser.add_argument("--explore-interval", type=int, default=20)
    parser.add_argument("--explore-extra", type=int, default=1)
    parser.add_argument("--include-rel-cons-in-main", action="store_true")

    parser.add_argument("--cache-dir", default="kb-cache")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-fallback-cache-dir", default=None)
    parser.add_argument("--exclude-source-domains", default="")
    parser.add_argument(
        "--no-kb-cache",
        action="store_false",
        dest="use_kb_cache",
        help="Ignore cached source embeddings and rebuild feedback pools from datasets.",
    )
    parser.set_defaults(use_kb_cache=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_tasr_ragroute(args)

    details_path = os.path.join(args.output_dir, f"tasr_ragroute_details_{timestamp}.csv")
    windows_path = os.path.join(args.output_dir, f"tasr_ragroute_windows_{timestamp}.csv")
    aggregates_path = os.path.join(args.output_dir, f"tasr_ragroute_aggregates_{timestamp}.csv")
    summary_path = os.path.join(args.output_dir, f"tasr_ragroute_summary_{timestamp}.csv")
    window_summary_path = os.path.join(args.output_dir, f"tasr_ragroute_window_summary_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"tasr_ragroute_summary_{timestamp}.json")

    write_csv(details_path, result["details"])
    write_csv(windows_path, result["windows"])
    write_csv(aggregates_path, result["aggregates"])
    write_csv(summary_path, result["aggregate_summary"])
    write_csv(window_summary_path, result["window_summary"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": result["config"],
                "aggregate_summary": result["aggregate_summary"],
                "window_summary": result["window_summary"],
                "files": {
                    "details_csv": details_path,
                    "windows_csv": windows_path,
                    "aggregates_csv": aggregates_path,
                    "summary_csv": summary_path,
                    "window_summary_csv": window_summary_path,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTASR RAGRoute online evaluation completed.")
    print(f"  Details: {details_path}")
    print(f"  Windows: {windows_path}")
    print(f"  Aggregates: {aggregates_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Window summary CSV: {window_summary_path}")
    print(f"  Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
