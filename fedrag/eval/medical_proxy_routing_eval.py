"""
Medical-query routing stress test for profile forgery.

This is a routing-level sanity check, not a full clinical FedRAG deployment.
It asks whether malicious clients with non-overlapping medical proxy profiles
can attract MedQA-USMLE test questions, even when honest medical clients are
present in the routing pool.
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
    compute_profile,
    compute_similarity_with_profile,
    embed_queries,
    embed_texts,
    get_embedding_dim,
    load_embedder,
)


STACKEXCHANGE_DOMAINS = [
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

DEFAULT_NONMEDICAL_DOMAINS = [
    "gaming",
    "gis",
    "physics",
    "electronics",
    "history",
    "law",
    "travel",
    "movies",
    "webapps",
    "android",
    "photo",
    "economics",
    "softwareengineering",
    "security",
    "cs",
]

STACKEXCHANGE_TEXT_CACHE: Dict[str, List[str]] = {}
MEDQA_CACHE: Dict[str, List[Dict]] = {}
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


def load_stackexchange_texts(domain: str) -> List[str]:
    if domain not in STACKEXCHANGE_TEXT_CACHE:
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
        STACKEXCHANGE_TEXT_CACHE[domain] = list(data["title_body"])
    return STACKEXCHANGE_TEXT_CACHE[domain]


def stackexchange_window(domain: str, n_docs: int, offset: int) -> List[str]:
    texts = load_stackexchange_texts(domain)
    if not texts or n_docs <= 0:
        return []
    start = int(offset) % len(texts)
    end = start + int(n_docs)
    if end <= len(texts):
        return texts[start:end]
    return texts[start:] + texts[: end - len(texts)]


def sample_stackexchange(domain: str, n_docs: int, offset: int, seed: int) -> List[str]:
    texts = load_stackexchange_texts(domain)
    if not texts or n_docs <= 0:
        return []
    start = min(max(int(offset), 0), len(texts))
    pool = texts[start:] or texts
    rng = np.random.RandomState(seed32(seed))
    replace = len(pool) < n_docs
    idx = rng.choice(len(pool), size=n_docs, replace=replace)
    return [pool[int(i)] for i in idx]


def load_medqa_split(split: str) -> List[Dict]:
    if split not in MEDQA_CACHE:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        MEDQA_CACHE[split] = [dict(item) for item in ds]
    return MEDQA_CACHE[split]


def medqa_passage(sample: Dict, include_answer: bool = True) -> str:
    options = sample.get("options") or {}
    opts = " ".join(f"{key}. {value}" for key, value in sorted(options.items()))
    parts = [f"Medical question: {sample.get('question', '')}", f"Options: {opts}"]
    if include_answer:
        answer = sample.get("answer", "")
        answer_idx = sample.get("answer_idx", "")
        parts.append(f"Correct answer: {answer_idx}. {answer}")
    phrases = sample.get("metamap_phrases") or []
    if phrases:
        parts.append("Medical concepts: " + ", ".join(str(x) for x in phrases[:16]))
    return " ".join(parts)


def medqa_query(sample: Dict, include_options: bool = False) -> str:
    if not include_options:
        return str(sample.get("question", ""))
    options = sample.get("options") or {}
    opts = " ".join(f"{key}. {value}" for key, value in sorted(options.items()))
    return f"{sample.get('question', '')} Options: {opts}"


def sample_indices(n: int, count: int, seed: int, offset: int = 0) -> List[int]:
    start = min(max(int(offset), 0), n)
    pool = list(range(start, n)) or list(range(n))
    rng = np.random.RandomState(seed32(seed))
    replace = len(pool) < count
    idx = rng.choice(len(pool), size=count, replace=replace)
    return [pool[int(i)] for i in idx]


def profile_from_docs(
    docs: List[str],
    profile_method: str,
    n_clusters: int,
    seed: int,
    sample_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_size and sample_size > 0 and len(docs) > sample_size:
        rng = np.random.RandomState(seed32(seed))
        idx = rng.choice(len(docs), size=sample_size, replace=False)
        docs = [docs[int(i)] for i in idx]
    embeddings = embed_texts(docs)
    profile_data = compute_profile(
        np.asarray(embeddings, dtype=np.float32),
        method=profile_method,
        n_clusters=n_clusters,
        seed=seed,
    )
    profile = normalize_rows(profile_data["profile"])
    if profile.ndim == 1:
        route_centroid = profile
    else:
        route_centroid = normalize_rows(profile.mean(axis=0))
    return route_centroid, profile


def profile_from_embeddings(
    embeddings: np.ndarray,
    profile_method: str,
    n_clusters: int,
    seed: int,
    sample_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if sample_size and sample_size > 0 and embeddings.shape[0] > sample_size:
        rng = np.random.RandomState(seed32(seed))
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[idx]
    profile_data = compute_profile(
        embeddings,
        method=profile_method,
        n_clusters=n_clusters,
        seed=seed,
    )
    profile = normalize_rows(profile_data["profile"])
    if profile.ndim == 1:
        route_centroid = profile
    else:
        route_centroid = normalize_rows(profile.mean(axis=0))
    return route_centroid, profile


def multidomain_cache_dir(cache_dir: str, pid: int, domains: List[str], domains_per_client: int) -> Tuple[str, List[str]]:
    rng = np.random.RandomState(seed32(pid))
    selected = rng.choice(
        domains,
        size=min(domains_per_client, len(domains)),
        replace=False,
    ).tolist()
    domain_str = "+".join(sorted(selected))
    return os.path.join(cache_dir, f"multi-{domain_str}-p{pid}"), sorted(selected)


def find_single_domain_cache(cache_dir: str, domain: str) -> Optional[str]:
    root = os.path.abspath(cache_dir)
    if not os.path.isdir(root):
        return None
    matches = []
    for name in os.listdir(root):
        if not name.startswith(f"{domain}-p"):
            continue
        path = os.path.join(root, name, domain, "embeddings.npy")
        if os.path.exists(path):
            matches.append(path)
    return sorted(matches)[0] if matches else None


def load_nonmedical_embeddings(pid: int, args: argparse.Namespace, domains: List[str]) -> Tuple[np.ndarray, List[str]]:
    cache_path, selected_domains = multidomain_cache_dir(
        args.cache_dir,
        pid,
        domains,
        args.domains_per_client,
    )
    embeddings_path = os.path.join(cache_path, "embeddings.npy")
    if args.use_kb_cache and os.path.exists(embeddings_path):
        return normalize_rows(np.load(embeddings_path)), selected_domains

    cached_parts = []
    if args.use_kb_cache:
        for domain in selected_domains:
            path = find_single_domain_cache(args.cache_dir, domain)
            if path:
                arr = np.load(path)
                if arr.ndim != 2 or arr.shape[1] != int(args.embedding_dim):
                    print(
                        f"[cache] Ignoring {path}: "
                        f"dim={arr.shape[1] if arr.ndim == 2 else 'invalid'}, "
                        f"expected={args.embedding_dim}"
                    )
                    continue
                if arr.shape[0] > args.nonmedical_docs_per_domain:
                    rng = np.random.RandomState(seed32(pid + stable_int(domain)))
                    idx = rng.choice(arr.shape[0], size=args.nonmedical_docs_per_domain, replace=False)
                    arr = arr[idx]
                cached_parts.append(arr)
        if len(cached_parts) == len(selected_domains):
            return normalize_rows(np.vstack(cached_parts)), selected_domains

    docs = []
    for domain in selected_domains:
        docs.extend(stackexchange_window(domain, args.nonmedical_docs_per_domain, args.stackexchange_offset))
    if not docs:
        raise RuntimeError(f"No documents loaded for nonmedical client {pid}")
    return normalize_rows(embed_texts(docs)), selected_domains


def build_base_clients(args: argparse.Namespace, nonmedical_domains: List[str]) -> Dict[int, Dict]:
    clients = {}

    for pid in tqdm(range(args.num_nonmedical_clients), desc="Non-medical clients"):
        embeddings, selected_domains = load_nonmedical_embeddings(pid, args, nonmedical_domains)
        route_centroid, profile = profile_from_embeddings(
            embeddings,
            args.profile_method,
            args.profile_n_clusters,
            seed=pid,
            sample_size=args.profile_sample_size,
        )
        clients[pid] = {
            "pid": pid,
            "kind": "honest_nonmedical",
            "domain": "+".join(selected_domains),
            "domains": selected_domains,
            "route_centroid": route_centroid,
            "profile": profile,
            "method": args.profile_method,
            "malicious": False,
        }

    train = load_medqa_split("train")
    reserved = args.medical_proxy_max + args.nonmedical_proxy_size
    available = len(train) - reserved
    if available < args.num_medical_clients * args.medical_docs_per_client:
        raise ValueError(
            "Not enough MedQA train samples for disjoint honest medical shards and proxy pools. "
            f"Available after reserved={reserved}: {available}"
        )

    start = 0
    for mid in tqdm(range(args.num_medical_clients), desc="Honest medical clients"):
        indices = list(range(start, start + args.medical_docs_per_client))
        start += args.medical_docs_per_client
        docs = [medqa_passage(train[i]) for i in indices]
        route_centroid, profile = profile_from_docs(
            docs,
            args.profile_method,
            args.profile_n_clusters,
            seed=10_000 + mid,
            sample_size=args.profile_sample_size,
        )
        pid = args.num_nonmedical_clients + mid
        clients[pid] = {
            "pid": pid,
            "kind": "honest_medical",
            "domain": "medqa_train_shard",
            "domains": ["medical"],
            "route_centroid": route_centroid,
            "profile": profile,
            "method": args.profile_method,
            "malicious": False,
        }

    return clients


def build_medqa_pools(args: argparse.Namespace) -> Dict[str, List[str]]:
    train = load_medqa_split("train")
    total_reserved = args.medical_proxy_max + args.nonmedical_proxy_size
    honest_span = args.num_medical_clients * args.medical_docs_per_client
    start = honest_span
    if start + total_reserved > len(train):
        raise ValueError("MedQA train split is too small for requested pool sizes")

    medical_proxy = [
        medqa_passage(train[i])
        for i in range(start, start + args.medical_proxy_max)
    ]
    start += args.medical_proxy_max

    nonmedical_docs = []
    domains = parse_str_list(args.nonmedical_domains)
    for i in range(args.nonmedical_proxy_size):
        domain = domains[i % len(domains)]
        nonmedical_docs.extend(
            sample_stackexchange(
                domain,
                1,
                args.stackexchange_offset + args.nonmedical_docs_per_domain,
                seed32(90_000 + i + stable_int(domain)),
            )
        )

    return {
        "medical_proxy": medical_proxy,
        "nonmedical_proxy": nonmedical_docs[: args.nonmedical_proxy_size],
    }


def build_malicious_profiles(
    base_pid: int,
    condition: str,
    proxy_size: int,
    pools: Dict[str, List[str]],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if condition == "random_profile":
        rng = np.random.RandomState(seed32(seed + base_pid * 997))
        dim = int(args.embedding_dim)
        if args.profile_method == "kmeans":
            n = max(1, args.profile_n_clusters)
            profile = rng.normal(size=(n, dim)).astype(np.float32)
            profile = normalize_rows(profile)
            route_centroid = normalize_rows(profile.mean(axis=0))
        else:
            profile = normalize_rows(rng.normal(size=dim).astype(np.float32))
            route_centroid = profile
        return route_centroid, profile, 0

    if condition == "medical_forged":
        source_docs = pools["medical_proxy"]
        n_medical_proxy = proxy_size
    elif condition == "nonmedical_forged":
        source_docs = pools["nonmedical_proxy"]
        n_medical_proxy = 0
    else:
        raise ValueError(f"Unknown malicious profile condition: {condition}")

    if len(source_docs) < proxy_size:
        raise ValueError(f"Condition {condition} only has {len(source_docs)} docs, need {proxy_size}")

    rng = np.random.RandomState(seed32(seed + base_pid * 997))
    idx = rng.choice(len(source_docs), size=proxy_size, replace=False)
    docs = [source_docs[int(i)] for i in idx]
    route_centroid, profile = profile_from_docs(
        docs,
        args.profile_method,
        args.profile_n_clusters,
        seed=seed32(seed + base_pid * 101),
        sample_size=0,
    )
    return route_centroid, profile, n_medical_proxy


def route_profiles(
    profiles: Dict[int, Dict],
    query_emb: np.ndarray,
) -> List[Tuple[int, float]]:
    scored = []
    for pid, profile_data in profiles.items():
        score = compute_similarity_with_profile(
            query_emb,
            profile_data["profile"],
            profile_data["method"],
        )
        scored.append((pid, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def any_in_topk(ranked: List[int], ids: Iterable[int], k: int) -> bool:
    target = set(ids)
    return any(pid in target for pid in ranked[:k])


def best_rank(ranked: List[int], ids: Iterable[int]) -> int:
    target = set(ids)
    for idx, pid in enumerate(ranked, start=1):
        if pid in target:
            return idx
    return len(ranked) + 1


def evaluate_condition(
    profiles: Dict[int, Dict],
    query_embeddings: np.ndarray,
    malicious_pids: List[int],
    honest_medical_pids: List[int],
    topk_list: List[int],
) -> Dict[str, float]:
    counts = {f"hr@{k}": 0 for k in topk_list}
    med_counts = {f"medacc@{k}": 0 for k in topk_list}
    mal_ranks = []
    med_ranks = []
    margins = []

    for qemb in query_embeddings:
        ranked_scores = route_profiles(profiles, qemb)
        ranked = [pid for pid, _ in ranked_scores]
        score_map = dict(ranked_scores)
        for k in topk_list:
            counts[f"hr@{k}"] += int(any_in_topk(ranked, malicious_pids, k))
            med_counts[f"medacc@{k}"] += int(any_in_topk(ranked, honest_medical_pids, k))
        mal_ranks.append(best_rank(ranked, malicious_pids))
        med_ranks.append(best_rank(ranked, honest_medical_pids))
        best_mal = max(score_map[pid] for pid in malicious_pids)
        best_med = max(score_map[pid] for pid in honest_medical_pids)
        margins.append(best_mal - best_med)

    n = max(len(query_embeddings), 1)
    metrics = {}
    for key, value in counts.items():
        metrics[key] = value / n
    for key, value in med_counts.items():
        metrics[key] = value / n
    metrics["mean_malicious_rank"] = mean(mal_ranks) if mal_ranks else 0.0
    metrics["mean_honest_medical_rank"] = mean(med_ranks) if med_ranks else 0.0
    metrics["mean_score_margin_mal_minus_med"] = mean(margins) if margins else 0.0
    return metrics


def aggregate_rows(rows: List[Dict], metric_keys: List[str]) -> List[Dict]:
    group_keys = [
        "condition",
        "proxy_size",
        "num_malicious",
        "profile_method",
    ]
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


def run_medical_proxy_routing(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    global DATASET_CACHE_DIR, DATASET_FALLBACK_CACHE_DIR, DATASET_REDOWNLOAD_CACHE_DIR

    DATASET_CACHE_DIR = args.dataset_cache_dir or None
    DATASET_FALLBACK_CACHE_DIR = (
        args.dataset_fallback_cache_dir
        or os.path.join(args.output_dir, "hf_datasets_cache_medical_proxy")
    )
    DATASET_REDOWNLOAD_CACHE_DIR = os.path.join(
        args.output_dir,
        f"hf_datasets_redownload_medical_proxy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    seeds = parse_int_list(args.seeds)
    topk_list = parse_int_list(args.topk_list)
    proxy_sizes = parse_int_list(args.proxy_sizes)
    num_malicious_list = parse_int_list(args.num_malicious_list)
    conditions = parse_str_list(args.conditions)
    nonmedical_domains = parse_str_list(args.nonmedical_domains)

    args.medical_proxy_max = max(proxy_sizes)
    args.nonmedical_proxy_size = max(proxy_sizes)

    try:
        load_embedder(args.emb_model, model_type=args.emb_model_type)
    except ImportError as exc:
        raise SystemExit(
            "Missing embedding dependencies. Use --emb-model-type sentence-transformer "
            "if FlagEmbedding is unavailable. Original error: "
            f"{exc}"
        ) from exc
    args.embedding_dim = int(get_embedding_dim())

    print("Loading MedQA-USMLE train/test splits")
    train = load_medqa_split("train")
    test = load_medqa_split("test")
    print(f"  train={len(train)}, test={len(test)}")

    base_clients = build_base_clients(args, nonmedical_domains)
    pools = build_medqa_pools(args)

    test_indices = sample_indices(len(test), args.num_queries, args.seed_base, offset=args.query_offset)
    query_texts = [medqa_query(test[i], include_options=args.query_include_options) for i in test_indices]
    print(f"Encoding {len(query_texts)} MedQA test queries")
    query_embeddings = normalize_rows(embed_queries(query_texts))
    if int(query_embeddings.shape[1]) != int(args.embedding_dim):
        raise ValueError(
            f"Query embedding dim mismatch: {query_embeddings.shape[1]} vs {args.embedding_dim}"
        )

    honest_medical_pids = [
        pid for pid, profile in base_clients.items()
        if profile["kind"] == "honest_medical"
    ]
    nonmedical_pids = [
        pid for pid, profile in base_clients.items()
        if profile["kind"] == "honest_nonmedical"
    ]

    rows = []
    for num_malicious in num_malicious_list:
        for seed in seeds:
            rng = np.random.RandomState(seed32(seed + 1701))
            chosen_nonmedical = rng.choice(
                nonmedical_pids,
                size=min(num_malicious, len(nonmedical_pids)),
                replace=False,
            ).tolist()
            malicious_pids = [
                args.num_nonmedical_clients + args.num_medical_clients + i
                for i in range(num_malicious)
            ]

            for proxy_size in proxy_sizes:
                for condition in conditions:
                    profiles = {
                        pid: dict(profile)
                        for pid, profile in base_clients.items()
                    }
                    total_medical_proxy = 0
                    for idx, mal_pid in enumerate(malicious_pids):
                        base_pid = int(chosen_nonmedical[idx % len(chosen_nonmedical)])
                        route_centroid, forged_profile, n_med_proxy = build_malicious_profiles(
                            base_pid,
                            condition,
                            proxy_size,
                            pools,
                            args,
                            seed32(seed + proxy_size * 13 + stable_int(condition)),
                        )
                        total_medical_proxy += n_med_proxy
                        profiles[mal_pid] = {
                            "pid": mal_pid,
                            "kind": "malicious",
                            "domain": f"malicious-{condition}",
                            "domains": [condition],
                            "route_centroid": route_centroid,
                            "profile": forged_profile,
                            "method": args.profile_method,
                            "malicious": True,
                            "actual_base_pid": base_pid,
                            "actual_base_domain": profiles[base_pid]["domain"],
                        }

                    metrics = evaluate_condition(
                        profiles,
                        query_embeddings,
                        malicious_pids,
                        honest_medical_pids,
                        topk_list,
                    )
                    row = {
                        "experiment": "medical_proxy_routing",
                        "condition": condition,
                        "proxy_size": proxy_size,
                        "num_malicious": num_malicious,
                        "seed": seed,
                        "profile_method": args.profile_method,
                        "profile_n_clusters": args.profile_n_clusters,
                        "num_queries": len(query_embeddings),
                        "num_nonmedical_clients": args.num_nonmedical_clients,
                        "num_honest_medical_clients": args.num_medical_clients,
                        "num_total_profiles": len(profiles),
                        "malicious_pids": " ".join(str(x) for x in malicious_pids),
                        "honest_medical_pids": " ".join(str(x) for x in honest_medical_pids),
                        "n_medical_proxy_docs": total_medical_proxy,
                    }
                    row.update(metrics)
                    rows.append(row)

    metric_keys = [f"hr@{k}" for k in topk_list]
    metric_keys += [f"medacc@{k}" for k in topk_list]
    metric_keys += [
        "mean_malicious_rank",
        "mean_honest_medical_rank",
        "mean_score_margin_mal_minus_med",
    ]
    summary_rows = aggregate_rows(rows, metric_keys)

    config = {
        "emb_model": args.emb_model,
        "emb_model_type": args.emb_model_type,
        "router": "embedding_cosine",
        "dataset": "GBaker/MedQA-USMLE-4-options",
        "target_queries": "MedQA test question stem",
        "medical_proxy_source": "MedQA train question/options/answer passages",
        "train_size": len(train),
        "test_size": len(test),
        "non_overlap": "MedQA train is used for honest medical/proxy profiles; test is used for queries.",
        "num_nonmedical_clients": args.num_nonmedical_clients,
        "num_medical_clients": args.num_medical_clients,
        "num_malicious_list": num_malicious_list,
        "num_queries": args.num_queries,
        "query_offset": args.query_offset,
        "proxy_sizes": proxy_sizes,
        "conditions": conditions,
        "topk_list": topk_list,
        "seeds": seeds,
        "profile_method": args.profile_method,
        "profile_n_clusters": args.profile_n_clusters,
        "nonmedical_domains": nonmedical_domains,
        "dataset_cache_dir": DATASET_CACHE_DIR,
        "dataset_fallback_cache_dir": DATASET_FALLBACK_CACHE_DIR,
        "dataset_redownload_cache_dir": DATASET_REDOWNLOAD_CACHE_DIR,
    }
    return rows, summary_rows, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical-query routing stress test")
    parser.add_argument("--num-nonmedical-clients", type=int, default=15)
    parser.add_argument("--num-medical-clients", type=int, default=3)
    parser.add_argument("--num-malicious-list", default="1,3")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-include-options", action="store_true")
    parser.add_argument("--proxy-sizes", default="25,50,100")
    parser.add_argument("--conditions", default="random_profile,nonmedical_forged,medical_forged")
    parser.add_argument("--topk-list", default="1,3,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--seed-base", type=int, default=2026)

    parser.add_argument("--medical-docs-per-client", type=int, default=500)
    parser.add_argument("--nonmedical-docs-per-domain", type=int, default=5000)
    parser.add_argument("--domains-per-client", type=int, default=3)
    parser.add_argument("--stackexchange-offset", type=int, default=0)
    parser.add_argument("--nonmedical-domains", default=",".join(DEFAULT_NONMEDICAL_DOMAINS))

    parser.add_argument("--profile-method", choices=["mean", "kmeans"], default="kmeans")
    parser.add_argument("--profile-n-clusters", type=int, default=5)
    parser.add_argument("--profile-sample-size", type=int, default=0)

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
        help="Ignore cached StackExchange embeddings and rebuild non-medical profiles from datasets.",
    )
    parser.set_defaults(use_kb_cache=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, summary_rows, config = run_medical_proxy_routing(args)

    details_path = os.path.join(args.output_dir, f"medical_proxy_routing_details_{timestamp}.csv")
    summary_csv_path = os.path.join(args.output_dir, f"medical_proxy_routing_summary_{timestamp}.csv")
    summary_json_path = os.path.join(args.output_dir, f"medical_proxy_routing_summary_{timestamp}.json")

    write_csv(details_path, rows)
    write_csv(summary_csv_path, summary_rows)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "summary": summary_rows,
                "details_csv": details_path,
                "summary_csv": summary_csv_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nMedical proxy routing stress test completed.")
    print(f"  Details: {details_path}")
    print(f"  Summary CSV: {summary_csv_path}")
    print(f"  Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
