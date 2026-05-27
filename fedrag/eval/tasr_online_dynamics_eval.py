"""
TASR warmup and online dynamics evaluation.

This script runs an offline embedding-routing simulation for the paper's
recurring-client FedRAG setting. It focuses on how TASR changes routing over a
query stream, rather than only reporting one final aggregate number.
"""

import argparse
import csv
import hashlib
import json
import os
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
    load_embedder,
)
from fedrag.rag.trust_defense import TrustAwareRouter  # noqa: E402


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
DOMAIN_LOAD_FAILURES: Dict[str, str] = {}
DATASET_CACHE_DIR: Optional[str] = None
DATASET_FALLBACK_CACHE_DIR: Optional[str] = None
DATASET_REDOWNLOAD_CACHE_DIR: Optional[str] = None
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
                        "fallback cache, and forced re-download attempts."
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


def select_malicious_pids(
    target_domain: str,
    num_malicious: int,
    seed: int,
    client_domains: Dict[int, List[str]],
) -> List[int]:
    candidates = [
        pid for pid, domains in client_domains.items()
        if target_domain not in domains
    ]
    if not candidates:
        raise ValueError(f"No eligible malicious clients for target={target_domain}")
    rng = np.random.RandomState(seed32(seed))
    chosen = rng.choice(
        candidates,
        size=min(num_malicious, len(candidates)),
        replace=False,
    )
    return sorted(int(x) for x in chosen)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        norm = np.linalg.norm(x) + 1e-8
        return (x / norm).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


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


def build_honest_clients(args: argparse.Namespace, domains: List[str]) -> Tuple[Dict[int, Dict], Dict[int, List[str]]]:
    clients = {}
    client_domains = {}
    for pid in tqdm(range(args.num_clients), desc="Building honest clients"):
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

        embeddings = np.asarray(embeddings, dtype=np.float32)
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
            "domain": "+".join(selected_domains),
            "route_centroid": route_centroid,
            "profile": profile,
            "doc_embeddings": normalize_rows(feedback_embeddings),
            "malicious": False,
            "source": source,
        }
        client_domains[pid] = selected_domains
    return clients, client_domains


def build_forged_malicious_client(
    base_client: Dict,
    target_domain: str,
    proxy_size: int,
    proxy_offset: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict:
    proxy_docs = sample_domain(target_domain, proxy_size, proxy_offset, seed)
    proxy_emb = embed_texts(proxy_docs)
    route_centroid, forged_profile = profile_from_embeddings(
        proxy_emb,
        args.profile_method,
        args.profile_n_clusters,
        seed=seed,
        sample_size=0,
    )
    forged = dict(base_client)
    forged["domain"] = f"malicious-{target_domain}"
    forged["domains"] = [f"malicious-{target_domain}"]
    forged["route_centroid"] = route_centroid
    forged["profile"] = forged_profile
    forged["malicious"] = True
    return forged


class ProfileTrustRouter(TrustAwareRouter):
    """TrustAwareRouter variant whose routing score uses K-Means profiles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_profiles: Dict[int, np.ndarray] = {}
        self.routing_methods: Dict[int, str] = {}

    def register_profile_client(
        self,
        client_id: int,
        route_centroid: np.ndarray,
        profile: np.ndarray,
        doc_embeddings: np.ndarray,
        method: str,
    ) -> None:
        self.register_client(
            client_id,
            route_centroid,
            doc_embeddings,
            profile_centroids=profile,
        )
        self.routing_profiles[client_id] = normalize_rows(profile)
        self.routing_methods[client_id] = method

    def route(self, query_emb: np.ndarray, top_k: int = 3):
        raw_scores = {}
        weighted_scores = {}
        for cid, profile in self.routing_profiles.items():
            raw = compute_similarity_with_profile(
                query_emb,
                profile,
                self.routing_methods[cid],
            )
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
            key=lambda x: weighted_scores[x],
            reverse=True,
        )
        selected = sorted_clients[:top_k]
        if (
            self.explore_interval > 0
            and self.query_count > 0
            and self.query_count % self.explore_interval == 0
        ):
            remaining = [c for c in sorted_clients[top_k:] if c not in selected]
            if remaining:
                selected = selected + remaining[:self.explore_extra]
        return selected, raw_scores


def build_router(
    clients: Dict[int, Dict],
    mode: str,
    args: argparse.Namespace,
    warmup_queries: int,
    cold_start_s0: float,
) -> ProfileTrustRouter:
    router = ProfileTrustRouter(
        decay_factor=args.decay_factor,
        recovery_factor=args.recovery_factor,
        min_reputation=args.min_reputation,
        warmup_queries=warmup_queries,
        cold_start_s0=cold_start_s0,
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
    for pid, client in clients.items():
        router.register_profile_client(
            pid,
            client["route_centroid"],
            client["profile"],
            client["doc_embeddings"],
            args.profile_method,
        )
    return router


def make_queries(target_domain: str, query_offset: int, num_queries: int) -> List[str]:
    texts = load_domain_texts(target_domain)
    if not texts:
        return []
    start = min(max(query_offset, 0), len(texts))
    pool = texts[start:] or texts
    if len(pool) >= num_queries:
        return pool[:num_queries]
    reps = (num_queries + len(pool) - 1) // len(pool)
    return (pool * reps)[:num_queries]


def is_hijack(selected: List[int], malicious_pids: Iterable[int], k: int) -> bool:
    malicious = set(malicious_pids)
    return any(pid in malicious for pid in selected[:k])


def is_correct(selected: List[int], target_clients: Iterable[int], k: int) -> bool:
    targets = set(target_clients)
    return any(pid in targets for pid in selected[:k])


def avg_dict_values(values: Dict[int, float], ids: Iterable[int]) -> float:
    ids = list(ids)
    if not ids:
        return 0.0
    present = [float(values[i]) for i in ids if i in values]
    return mean(present) if present else 0.0


def avg_effective(router: ProfileTrustRouter, ids: Iterable[int], top_k: int) -> float:
    ids = list(ids)
    vals = [router.get_effective_score(i, top_k=top_k) for i in ids if i in router.reputation]
    return mean(vals) if vals else 0.0


def run_stream(
    clients: Dict[int, Dict],
    query_embeddings: np.ndarray,
    target_clients: List[int],
    malicious_pids: List[int],
    target_domain: str,
    num_malicious: int,
    seed: int,
    mode_specs: List[Tuple[str, str]],
    args: argparse.Namespace,
    warmup_queries: int,
    cold_start_s0: float,
    experiment_label: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    routers = {}
    for label, mode in mode_specs:
        routers[label] = build_router(
            clients,
            mode,
            args,
            warmup_queries=warmup_queries,
            cold_start_s0=cold_start_s0,
        )

    detail_rows = []
    window_rows = []
    aggregate_rows = []
    running = {
        label: {
            "h1": 0,
            "h3": 0,
            "a1": 0,
            "a3": 0,
            "window": [],
        }
        for label, _ in mode_specs
    }

    for qi, qemb in enumerate(query_embeddings, start=1):
        for label, _ in mode_specs:
            router = routers[label]
            selected, scores = router.route(qemb, top_k=args.route_top_k)
            row = {
                "experiment": experiment_label,
                "target_domain": target_domain,
                "num_malicious": num_malicious,
                "seed": seed,
                "mode": label,
                "warmup_queries": warmup_queries,
                "cold_start_s0": cold_start_s0,
                "query_idx": qi,
                "window_idx": (qi - 1) // args.window_size + 1,
                "selected": " ".join(str(x) for x in selected),
                "top1": selected[0] if selected else -1,
                "hijack@1": int(is_hijack(selected, malicious_pids, 1)),
                "hijack@3": int(is_hijack(selected, malicious_pids, min(3, args.route_top_k))),
                "acc@1": int(is_correct(selected, target_clients, 1)),
                "acc@3": int(is_correct(selected, target_clients, min(3, args.route_top_k))),
            }
            detail_rows.append(row)

            running[label]["h1"] += row["hijack@1"]
            running[label]["h3"] += row["hijack@3"]
            running[label]["a1"] += row["acc@1"]
            running[label]["a3"] += row["acc@3"]
            running[label]["window"].append(row)

            if label != "No Defense":
                router.update_trust(qemb, selected)

        if qi % args.window_size == 0 or qi == len(query_embeddings):
            for label, _ in mode_specs:
                router = routers[label]
                wrows = running[label]["window"]
                wcount = len(wrows)
                h1 = sum(r["hijack@1"] for r in wrows) / max(wcount, 1)
                h3 = sum(r["hijack@3"] for r in wrows) / max(wcount, 1)
                a1 = sum(r["acc@1"] for r in wrows) / max(wcount, 1)
                a3 = sum(r["acc@3"] for r in wrows) / max(wcount, 1)
                honest_ids = [pid for pid in clients if pid not in malicious_pids]
                window_rows.append({
                    "experiment": experiment_label,
                    "target_domain": target_domain,
                    "num_malicious": num_malicious,
                    "seed": seed,
                    "mode": label,
                    "warmup_queries": warmup_queries,
                    "cold_start_s0": cold_start_s0,
                    "window_idx": (qi - 1) // args.window_size + 1,
                    "query_start": qi - wcount + 1,
                    "query_end": qi,
                    "window_size": wcount,
                    "window_hr@1": h1,
                    "window_hr@3": h3,
                    "window_acc@1": a1,
                    "window_acc@3": a3,
                    "cumulative_hr@1": running[label]["h1"] / qi,
                    "cumulative_hr@3": running[label]["h3"] / qi,
                    "cumulative_acc@1": running[label]["a1"] / qi,
                    "cumulative_acc@3": running[label]["a3"] / qi,
                    "mal_reputation": avg_dict_values(router.reputation, malicious_pids),
                    "honest_reputation": avg_dict_values(router.reputation, honest_ids),
                    "honest_target_reputation": avg_dict_values(router.reputation, target_clients),
                    "mal_consistency": avg_dict_values(router.consistency_trust, malicious_pids),
                    "honest_target_consistency": avg_dict_values(router.consistency_trust, target_clients),
                    "mal_agreement": avg_dict_values(router.agreement_trust, malicious_pids),
                    "honest_target_agreement": avg_dict_values(router.agreement_trust, target_clients),
                    "mal_effective_trust": avg_effective(router, malicious_pids, args.route_top_k),
                    "honest_target_effective_trust": avg_effective(router, target_clients, args.route_top_k),
                    "mal_cold_start": mean([
                        router._cold_start_factor(pid)
                        for pid in malicious_pids
                    ]) if malicious_pids else 0.0,
                })
                running[label]["window"] = []

    for label, _ in mode_specs:
        router = routers[label]
        late_start = max(0, len(query_embeddings) - args.late_queries)
        early_end = min(len(query_embeddings), args.early_queries)
        label_rows = [r for r in detail_rows if r["mode"] == label]
        early_rows = label_rows[:early_end]
        late_rows = label_rows[late_start:]
        aggregate_rows.append({
            "experiment": experiment_label,
            "target_domain": target_domain,
            "num_malicious": num_malicious,
            "seed": seed,
            "mode": label,
            "warmup_queries": warmup_queries,
            "cold_start_s0": cold_start_s0,
            "total_queries": len(query_embeddings),
            "final_hr@1": running[label]["h1"] / len(query_embeddings),
            "final_hr@3": running[label]["h3"] / len(query_embeddings),
            "final_acc@1": running[label]["a1"] / len(query_embeddings),
            "final_acc@3": running[label]["a3"] / len(query_embeddings),
            "early_hr@1": mean([r["hijack@1"] for r in early_rows]) if early_rows else 0.0,
            "early_acc@1": mean([r["acc@1"] for r in early_rows]) if early_rows else 0.0,
            "late_hr@1": mean([r["hijack@1"] for r in late_rows]) if late_rows else 0.0,
            "late_acc@1": mean([r["acc@1"] for r in late_rows]) if late_rows else 0.0,
            "final_mal_reputation": avg_dict_values(router.reputation, malicious_pids),
            "final_honest_target_reputation": avg_dict_values(router.reputation, target_clients),
            "final_mal_effective_trust": avg_effective(router, malicious_pids, args.route_top_k),
            "final_honest_target_effective_trust": avg_effective(router, target_clients, args.route_top_k),
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


def run_tasr_online(args: argparse.Namespace) -> Dict:
    global DATASET_CACHE_DIR, DATASET_FALLBACK_CACHE_DIR, DATASET_REDOWNLOAD_CACHE_DIR

    domains = parse_str_list(args.domains)
    targets = parse_str_list(args.target_domains)
    seeds = parse_int_list(args.seeds)
    num_malicious_list = parse_int_list(args.num_malicious_list)
    warmup_values = parse_int_list(args.warmup_values)
    cold_start_values = parse_float_list(args.cold_start_values)
    excluded_noise_domains = parse_str_list(args.exclude_noise_domains)
    DOMAIN_LOAD_FAILURES.update({domain: "excluded" for domain in excluded_noise_domains})

    DATASET_CACHE_DIR = args.dataset_cache_dir or None
    DATASET_FALLBACK_CACHE_DIR = (
        args.dataset_fallback_cache_dir
        or os.path.join(args.output_dir, "hf_datasets_cache")
    )
    DATASET_REDOWNLOAD_CACHE_DIR = os.path.join(
        args.output_dir,
        f"hf_datasets_redownload_tasr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    try:
        load_embedder(args.emb_model, model_type=args.emb_model_type)
    except ImportError as exc:
        raise SystemExit(
            "Missing embedding dependencies. Use the same environment as FedRAG "
            f"experiments. Original error: {exc}"
        ) from exc

    base_clients, client_domains = build_honest_clients(args, domains)

    mode_specs = [
        ("No Defense", "none"),
        ("Rel", "rel"),
        ("Rel+Cons", "rel_cons"),
        ("Full TASR", "rel_cons_agr"),
    ]
    main_mode_specs = [
        ("No Defense", "none"),
        ("Rel", "rel"),
        ("Full TASR", "rel_cons_agr"),
    ]
    if args.include_rel_cons_in_main:
        main_mode_specs = mode_specs

    all_details = []
    all_windows = []
    all_aggregates = []

    query_embeddings_by_target = {}
    for target in targets:
        queries = make_queries(target, args.query_offset, args.num_queries)
        print(f"Encoding query stream for target={target} ({len(queries)} queries)")
        query_embeddings_by_target[target] = np.asarray(embed_queries(queries), dtype=np.float32)

    for target in targets:
        query_embeddings = query_embeddings_by_target[target]
        for num_malicious in num_malicious_list:
            for seed in seeds:
                malicious_pids = select_malicious_pids(
                    target,
                    num_malicious,
                    seed,
                    client_domains,
                )
                target_clients = [
                    pid for pid, ds in client_domains.items()
                    if pid not in malicious_pids and target in ds
                ]
                clients = {
                    pid: dict(client)
                    for pid, client in base_clients.items()
                }
                for pid in malicious_pids:
                    clients[pid] = build_forged_malicious_client(
                        clients[pid],
                        target,
                        args.proxy_size,
                        args.proxy_offset,
                        seed32(seed + pid * 1009 + stable_int(target)),
                        args,
                    )

                details, windows, aggregates = run_stream(
                    clients,
                    query_embeddings,
                    target_clients,
                    malicious_pids,
                    target,
                    num_malicious,
                    seed,
                    main_mode_specs,
                    args,
                    warmup_queries=args.warmup_queries,
                    cold_start_s0=args.cold_start_s0,
                    experiment_label="online",
                )
                all_details.extend(details)
                all_windows.extend(windows)
                all_aggregates.extend(aggregates)

                if args.run_warmup_sweep:
                    for warmup in warmup_values:
                        if warmup == args.warmup_queries:
                            continue
                        _, warm_windows, warm_aggregates = run_stream(
                            clients,
                            query_embeddings,
                            target_clients,
                            malicious_pids,
                            target,
                            num_malicious,
                            seed,
                            [("Full TASR", "rel_cons_agr")],
                            args,
                            warmup_queries=warmup,
                            cold_start_s0=args.cold_start_s0,
                            experiment_label="warmup_sweep",
                        )
                        all_windows.extend(warm_windows)
                        all_aggregates.extend(warm_aggregates)

                if args.run_cold_start_sweep:
                    for s0 in cold_start_values:
                        if abs(s0 - args.cold_start_s0) < 1e-12:
                            continue
                        _, s0_windows, s0_aggregates = run_stream(
                            clients,
                            query_embeddings,
                            target_clients,
                            malicious_pids,
                            target,
                            num_malicious,
                            seed,
                            [("Full TASR", "rel_cons_agr")],
                            args,
                            warmup_queries=args.warmup_queries,
                            cold_start_s0=s0,
                            experiment_label="cold_start_sweep",
                        )
                        all_windows.extend(s0_windows)
                        all_aggregates.extend(s0_aggregates)

    aggregate_metrics = [
        "final_hr@1",
        "final_hr@3",
        "final_acc@1",
        "final_acc@3",
        "early_hr@1",
        "early_acc@1",
        "late_hr@1",
        "late_acc@1",
        "final_mal_reputation",
        "final_honest_target_reputation",
        "final_mal_effective_trust",
        "final_honest_target_effective_trust",
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
            "cumulative_hr@1",
            "cumulative_acc@1",
            "mal_effective_trust",
            "honest_target_effective_trust",
            "mal_reputation",
            "honest_target_reputation",
        ],
    )

    config = {
        "emb_model": args.emb_model,
        "router": "embedding_tasr",
        "client_environment": f"{args.num_clients}-client StackExchange FedRAG",
        "topology": "multi-domain",
        "profile_method": args.profile_method,
        "profile_n_clusters": args.profile_n_clusters,
        "target_domains": targets,
        "num_malicious_list": num_malicious_list,
        "seeds": seeds,
        "num_queries": args.num_queries,
        "window_size": args.window_size,
        "route_top_k": args.route_top_k,
        "proxy_size": args.proxy_size,
        "warmup_queries": args.warmup_queries,
        "warmup_values": warmup_values,
        "cold_start_s0": args.cold_start_s0,
        "cold_start_values": cold_start_values,
        "decay_factor": args.decay_factor,
        "recovery_factor": args.recovery_factor,
        "exclude_noise_domains": excluded_noise_domains,
        "dataset_cache_dir": DATASET_CACHE_DIR,
        "dataset_fallback_cache_dir": DATASET_FALLBACK_CACHE_DIR,
        "dataset_redownload_cache_dir": DATASET_REDOWNLOAD_CACHE_DIR,
    }

    return {
        "config": config,
        "details": all_details,
        "windows": all_windows,
        "aggregates": all_aggregates,
        "aggregate_summary": aggregate_summary,
        "window_summary": window_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TASR online dynamics evaluation")

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
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--domains", default=",".join(DOMAIN_LIST))
    parser.add_argument("--domains-per-client", type=int, default=3)
    parser.add_argument("--honest-docs-per-domain", type=int, default=10000)
    parser.add_argument("--feedback-docs-per-client", type=int, default=500)
    parser.add_argument("--feedback-docs", type=int, default=5)

    parser.add_argument("--profile-method", choices=["mean", "kmeans"], default="kmeans")
    parser.add_argument("--profile-n-clusters", type=int, default=5)
    parser.add_argument("--profile-sample-size", type=int, default=0)

    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--warmup-values", default="0,25,50,100")
    parser.add_argument("--cold-start-s0", type=float, default=0.7)
    parser.add_argument("--cold-start-values", default="0.5,0.7,0.9,1.0")
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

    parser.add_argument("--run-warmup-sweep", action="store_true")
    parser.add_argument("--run-cold-start-sweep", action="store_true")
    parser.add_argument("--include-rel-cons-in-main", action="store_true")

    parser.add_argument("--emb-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--emb-model-type", default="sentence-transformer")
    parser.add_argument("--cache-dir", default="kb-cache")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-fallback-cache-dir", default=None)
    parser.add_argument("--exclude-noise-domains", default="")
    parser.add_argument(
        "--no-kb-cache",
        action="store_false",
        dest="use_kb_cache",
        help="Ignore existing kb-cache embeddings and rebuild honest embeddings from datasets.",
    )
    parser.set_defaults(use_kb_cache=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_tasr_online(args)

    details_path = os.path.join(args.output_dir, f"tasr_online_details_{timestamp}.csv")
    windows_path = os.path.join(args.output_dir, f"tasr_online_windows_{timestamp}.csv")
    aggregates_path = os.path.join(args.output_dir, f"tasr_online_aggregates_{timestamp}.csv")
    sensitivity_path = os.path.join(args.output_dir, f"tasr_online_sensitivity_{timestamp}.csv")
    aggregate_summary_path = os.path.join(args.output_dir, f"tasr_online_summary_{timestamp}.csv")
    window_summary_path = os.path.join(args.output_dir, f"tasr_online_window_summary_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"tasr_online_summary_{timestamp}.json")

    write_csv(details_path, result["details"])
    write_csv(windows_path, result["windows"])
    write_csv(aggregates_path, result["aggregates"])
    sensitivity_rows = [
        row for row in result["aggregate_summary"]
        if row["experiment"] in ("warmup_sweep", "cold_start_sweep")
    ]
    write_csv(sensitivity_path, sensitivity_rows)
    write_csv(aggregate_summary_path, result["aggregate_summary"])
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
                    "sensitivity_csv": sensitivity_path,
                    "aggregate_summary_csv": aggregate_summary_path,
                    "window_summary_csv": window_summary_path,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTASR online dynamics completed.")
    print(f"  Details: {details_path}")
    print(f"  Windows: {windows_path}")
    print(f"  Aggregates: {aggregates_path}")
    print(f"  Sensitivity: {sensitivity_path}")
    print(f"  Summary CSV: {aggregate_summary_path}")
    print(f"  Window summary CSV: {window_summary_path}")
    print(f"  Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
