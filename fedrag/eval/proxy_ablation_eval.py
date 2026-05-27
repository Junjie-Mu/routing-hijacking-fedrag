"""
Proxy-data scarcity and noise ablation for routing hijacking.

This script evaluates embedding-based routing under the default paper setting:
20-client StackExchange FedRAG, multi-domain clients, and K-Means profiles.
It does not start Flower. Instead, it builds the same profile-level routing
surface offline and measures how often forged malicious profiles enter Top-K.
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
DOMAIN_LOAD_FAILURES: Dict[str, str] = {}
DATASET_CACHE_DIR: Optional[str] = None
DATASET_FALLBACK_CACHE_DIR: Optional[str] = None
DATASET_REDOWNLOAD_CACHE_DIR: Optional[str] = None


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def seed32(value: int) -> int:
    return int(value) % (2**32 - 1)


DATASET_CACHE_ERRORS = (JSONDecodeError, DatasetGenerationError, OSError)


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
                f"{type(exc).__name__}. "
                f"retrying with fallback cache: {DATASET_FALLBACK_CACHE_DIR}"
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
                    f"{type(retry_exc).__name__}. "
                    f"forcing a fresh download into: {redownload_dir}"
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
                        "fallback cache, and forced re-download attempts. This usually "
                        "means the network/proxy returned a corrupted parquet file. "
                        "Try rerunning with a new --dataset-fallback-cache-dir, or "
                        "pre-download the dataset from a stable network."
                    ) from final_exc
            except Exception:
                raise
        except Exception:
            raise
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


def sample_nontarget(
    target_domain: str,
    n_docs: int,
    proxy_offset: int,
    seed: int,
    domains: List[str],
) -> List[str]:
    if n_docs <= 0:
        return []
    choices = [
        d
        for d in domains
        if d != target_domain and d not in DOMAIN_LOAD_FAILURES
    ]
    if not choices:
        raise ValueError("No non-target domains available for distractor sampling")

    rng = np.random.RandomState(seed32(seed))
    docs = []
    available = list(choices)

    while len(docs) < n_docs and available:
        remaining = n_docs - len(docs)
        picked = rng.choice(available, size=remaining, replace=True)
        made_progress = False

        for domain in list(available):
            count = int(np.sum(picked == domain))
            if not count:
                continue
            try:
                batch = sample_domain(
                    domain,
                    count,
                    proxy_offset,
                    seed32(seed + stable_int(domain) + len(docs)),
                )
            except Exception as exc:
                DOMAIN_LOAD_FAILURES[domain] = type(exc).__name__
                available.remove(domain)
                print(
                    f"[datasets] Skipping non-target distractor domain '{domain}' "
                    f"after load failure: {type(exc).__name__}"
                )
                continue

            if not batch:
                DOMAIN_LOAD_FAILURES[domain] = "empty"
                available.remove(domain)
                print(
                    f"[datasets] Skipping non-target distractor domain '{domain}' "
                    "because it returned no documents."
                )
                continue

            docs.extend(batch)
            made_progress = True

        if not made_progress and not available:
            break

    if len(docs) < n_docs:
        raise RuntimeError(
            f"Only sampled {len(docs)} / {n_docs} non-target proxy documents for "
            f"target={target_domain}. Failed domains: {DOMAIN_LOAD_FAILURES}"
        )

    docs = docs[:n_docs]
    rng.shuffle(docs)
    return docs


def build_proxy_docs(
    target_domain: str,
    total_size: int,
    target_fraction: float,
    proxy_offset: int,
    seed: int,
    domains: List[str],
) -> Tuple[List[str], int, int]:
    target_fraction = max(0.0, min(1.0, float(target_fraction)))
    n_target = int(round(total_size * target_fraction))
    n_noise = int(total_size - n_target)

    target_docs = sample_domain(target_domain, n_target, proxy_offset, seed)
    noise_docs = sample_nontarget(
        target_domain,
        n_noise,
        proxy_offset,
        seed + 100_000,
        domains,
    )

    docs = target_docs + noise_docs
    rng = np.random.RandomState(seed32(seed + 200_000))
    rng.shuffle(docs)
    return docs, n_target, n_noise


def select_malicious_pids(
    target_domain: str,
    num_malicious: int,
    seed: int,
    honest_profiles: Dict[int, Dict],
) -> List[int]:
    candidates = [
        pid
        for pid, profile in honest_profiles.items()
        if target_domain not in str(profile["domain"]).split("+")
    ]
    if not candidates:
        raise ValueError("No eligible malicious clients")
    rng = np.random.RandomState(seed32(seed))
    chosen = rng.choice(
        candidates,
        size=min(num_malicious, len(candidates)),
        replace=False,
    )
    return sorted(int(x) for x in chosen)


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
    return os.path.join(cache_dir, f"multi-{domain_str}-p{pid}"), selected


def profile_from_embeddings(
    embeddings: np.ndarray,
    method: str,
    n_clusters: int,
    seed: int,
    sample_size: int,
) -> np.ndarray:
    if sample_size and sample_size > 0 and embeddings.shape[0] > sample_size:
        rng = np.random.RandomState(seed32(seed))
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[idx]
    return compute_profile(
        embeddings,
        method=method,
        n_clusters=n_clusters,
        seed=seed,
    )["profile"]


def build_honest_profiles(args: argparse.Namespace, domains: List[str]) -> Dict[int, Dict]:
    profiles = {}
    iterator = tqdm(range(args.num_clients), desc="Building honest profiles")
    for pid in iterator:
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

        profile = profile_from_embeddings(
            np.asarray(embeddings, dtype=np.float32),
            args.profile_method,
            args.profile_n_clusters,
            seed=pid,
            sample_size=args.profile_sample_size,
        )
        profiles[pid] = {
            "pid": pid,
            "domain": "+".join(sorted(selected_domains)),
            "malicious": False,
            "profile": profile,
            "method": args.profile_method,
            "source": source,
        }
    return profiles


def make_malicious_profile(
    proxy_docs: List[str],
    method: str,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    if not proxy_docs:
        raise ValueError("proxy_docs is empty")
    embeddings = embed_texts(proxy_docs)
    return compute_profile(
        np.asarray(embeddings, dtype=np.float32),
        method=method,
        n_clusters=n_clusters,
        seed=seed,
    )["profile"]


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


def evaluate_hijack(
    profiles: Dict[int, Dict],
    query_embeddings: np.ndarray,
    malicious_pids: Iterable[int],
    topk_list: List[int],
) -> Dict[str, float]:
    malicious_set = set(malicious_pids)
    max_k = max(topk_list)
    counts = {k: 0 for k in topk_list}
    ranks = []
    margins = []

    profile_items = list(profiles.values())
    for qemb in query_embeddings:
        scored = []
        for profile in profile_items:
            score = compute_similarity_with_profile(
                qemb,
                profile["profile"],
                profile["method"],
            )
            scored.append((score, profile["pid"], profile["malicious"]))

        scored.sort(key=lambda x: -x[0])
        top = scored[:max_k]
        for k in topk_list:
            if any(pid in malicious_set for _, pid, _ in top[:k]):
                counts[k] += 1

        for rank, (_, pid, _) in enumerate(scored, start=1):
            if pid in malicious_set:
                ranks.append(rank)
                break

        best_mal = max(score for score, pid, _ in scored if pid in malicious_set)
        best_honest = max(score for score, pid, _ in scored if pid not in malicious_set)
        margins.append(best_mal - best_honest)

    total = float(len(query_embeddings))
    result = {f"hr@{k}": counts[k] / total if total else 0.0 for k in topk_list}
    result["mean_malicious_rank"] = mean(ranks) if ranks else 0.0
    result["mean_score_margin"] = mean(margins) if margins else 0.0
    return result


def apply_malicious_profiles(
    honest_profiles: Dict[int, Dict],
    forged_profiles: Dict[int, np.ndarray],
    method: str,
    condition_label: str,
) -> Dict[int, Dict]:
    profiles = {
        pid: {
            **profile,
            "profile": np.array(profile["profile"], copy=True),
            "malicious": False,
        }
        for pid, profile in honest_profiles.items()
    }
    for pid, forged_profile in forged_profiles.items():
        profiles[pid] = {
            **profiles[pid],
            "domain": condition_label,
            "malicious": True,
            "profile": np.array(forged_profile, copy=True),
            "method": method,
        }
    return profiles


def build_forged_profiles(
    target_domain: str,
    malicious_pids: List[int],
    proxy_size: int,
    target_fraction: float,
    proxy_offset: int,
    base_seed: int,
    domains: List[str],
    method: str,
    n_clusters: int,
) -> Tuple[Dict[int, np.ndarray], int, int]:
    forged_profiles = {}
    n_target = 0
    n_noise = 0
    for pid in malicious_pids:
        docs, n_target, n_noise = build_proxy_docs(
            target_domain,
            proxy_size,
            target_fraction,
            proxy_offset,
            seed32(base_seed + pid * 1009),
            domains,
        )
        forged_profiles[pid] = make_malicious_profile(
            docs,
            method,
            n_clusters,
            seed32(base_seed + pid * 9176),
        )
    return forged_profiles, n_target, n_noise


def row_for_condition(
    experiment: str,
    target_domain: str,
    num_malicious: int,
    seed: int,
    malicious_pids: List[int],
    proxy_condition: str,
    proxy_size: int,
    target_fraction: float,
    n_target_proxy: int,
    n_noise_proxy: int,
    metrics: Dict[str, float],
) -> Dict:
    row = {
        "experiment": experiment,
        "target_domain": target_domain,
        "num_malicious": num_malicious,
        "seed": seed,
        "malicious_pids": " ".join(str(x) for x in malicious_pids),
        "proxy_condition": proxy_condition,
        "proxy_size": proxy_size,
        "target_fraction": target_fraction,
        "n_target_proxy": n_target_proxy,
        "n_noise_proxy": n_noise_proxy,
    }
    row.update(metrics)
    return row


def aggregate_rows(rows: List[Dict], metric_keys: List[str]) -> List[Dict]:
    group_keys = [
        "experiment",
        "target_domain",
        "num_malicious",
        "proxy_condition",
        "proxy_size",
        "target_fraction",
        "n_target_proxy",
        "n_noise_proxy",
    ]
    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, items in sorted(groups.items(), key=lambda x: x[0]):
        out = {k: v for k, v in zip(group_keys, key)}
        out["n_seeds"] = len(items)
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_proxy_ablation(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    global DATASET_CACHE_DIR, DATASET_FALLBACK_CACHE_DIR, DATASET_REDOWNLOAD_CACHE_DIR

    domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    excluded_noise_domains = {
        d.strip().lower()
        for d in args.exclude_noise_domains.split(",")
        if d.strip()
    }
    DOMAIN_LOAD_FAILURES.update({domain: "excluded" for domain in excluded_noise_domains})
    targets = [d.strip().lower() for d in args.target_domains.split(",") if d.strip()]
    seeds = parse_int_list(args.seeds)
    num_malicious_list = parse_int_list(args.num_malicious_list)
    proxy_sizes = parse_int_list(args.proxy_sizes)
    target_fractions = parse_float_list(args.target_fractions)
    topk_list = parse_int_list(args.topk_list)
    DATASET_CACHE_DIR = args.dataset_cache_dir or None
    DATASET_FALLBACK_CACHE_DIR = (
        args.dataset_fallback_cache_dir
        or os.path.join(args.output_dir, "hf_datasets_cache")
    )
    DATASET_REDOWNLOAD_CACHE_DIR = os.path.join(
        args.output_dir,
        f"hf_datasets_redownload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    try:
        load_embedder(args.emb_model, model_type=args.emb_model_type)
    except ImportError as exc:
        raise SystemExit(
            "Missing embedding dependencies in this Python environment. "
            "Run this script from the same environment used for the FedRAG "
            "experiments, or install the packages in requirements.txt. "
            f"Original error: {exc}"
        ) from exc

    honest_profiles = build_honest_profiles(args, domains)

    rows = []
    query_embeddings_by_target = {}
    for target in targets:
        print(f"\nEncoding evaluation queries for target={target}")
        queries = make_queries(target, args.query_offset, args.num_queries)
        query_embeddings_by_target[target] = np.asarray(embed_queries(queries), dtype=np.float32)

    for target in targets:
        query_embeddings = query_embeddings_by_target[target]
        for num_malicious in num_malicious_list:
            for seed in seeds:
                malicious_pids = select_malicious_pids(
                    target,
                    num_malicious,
                    seed,
                    honest_profiles,
                )

                if args.experiment in ("all", "scarcity"):
                    for proxy_size in proxy_sizes:
                        for condition, fraction in (
                            ("target_clean", 1.0),
                            ("random_nontarget", 0.0),
                        ):
                            forged_profiles, n_target, n_noise = build_forged_profiles(
                                target,
                                malicious_pids,
                                proxy_size,
                                fraction,
                                args.proxy_offset,
                                seed32(seed + proxy_size + stable_int(condition)),
                                domains,
                                args.profile_method,
                                args.profile_n_clusters,
                            )
                            profiles = apply_malicious_profiles(
                                honest_profiles,
                                forged_profiles,
                                args.profile_method,
                                f"malicious-{condition}-{target}",
                            )
                            metrics = evaluate_hijack(
                                profiles,
                                query_embeddings,
                                malicious_pids,
                                topk_list,
                            )
                            rows.append(
                                row_for_condition(
                                    "scarcity",
                                    target,
                                    num_malicious,
                                    seed,
                                    malicious_pids,
                                    condition,
                                    proxy_size,
                                    fraction,
                                    n_target,
                                    n_noise,
                                    metrics,
                                )
                            )

                if args.experiment in ("all", "noise"):
                    for fraction in target_fractions:
                        forged_profiles, n_target, n_noise = build_forged_profiles(
                            target,
                            malicious_pids,
                            args.total_proxy_size,
                            fraction,
                            args.proxy_offset,
                            seed32(seed + stable_int(f"noise-{fraction}")),
                            domains,
                            args.profile_method,
                            args.profile_n_clusters,
                        )
                        profiles = apply_malicious_profiles(
                            honest_profiles,
                            forged_profiles,
                            args.profile_method,
                            f"malicious-noise-{target}",
                        )
                        metrics = evaluate_hijack(
                            profiles,
                            query_embeddings,
                            malicious_pids,
                            topk_list,
                        )
                        rows.append(
                            row_for_condition(
                                "noise",
                                target,
                                num_malicious,
                                seed,
                                malicious_pids,
                                "target_fraction",
                                args.total_proxy_size,
                                fraction,
                                n_target,
                                n_noise,
                                metrics,
                            )
                        )

    metric_keys = [f"hr@{k}" for k in topk_list] + [
        "mean_malicious_rank",
        "mean_score_margin",
    ]
    summary_rows = aggregate_rows(rows, metric_keys)

    config = {
        "emb_model": args.emb_model,
        "router": "embedding",
        "client_environment": f"{args.num_clients}-client StackExchange FedRAG",
        "topology": "multi-domain",
        "profile_method": args.profile_method,
        "profile_n_clusters": args.profile_n_clusters,
        "num_clients": args.num_clients,
        "domains_per_client": args.domains_per_client,
        "targets": targets,
        "num_malicious_list": num_malicious_list,
        "seeds": seeds,
        "topk_list": topk_list,
        "proxy_offset": args.proxy_offset,
        "query_offset": args.query_offset,
        "num_queries": args.num_queries,
        "proxy_sizes": proxy_sizes,
        "total_proxy_size": args.total_proxy_size,
        "target_fractions": target_fractions,
        "use_kb_cache": args.use_kb_cache,
        "profile_sample_size": args.profile_sample_size,
        "dataset_cache_dir": DATASET_CACHE_DIR,
        "dataset_fallback_cache_dir": DATASET_FALLBACK_CACHE_DIR,
        "dataset_redownload_cache_dir": DATASET_REDOWNLOAD_CACHE_DIR,
        "exclude_noise_domains": sorted(excluded_noise_domains),
    }
    return rows, summary_rows, config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proxy-data scarcity and noise ablation for routing hijacking"
    )
    parser.add_argument("--experiment", choices=["all", "scarcity", "noise"], default="all")
    parser.add_argument("--target-domains", default="gaming,gis,physics")
    parser.add_argument("--num-malicious-list", default="1,3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--topk-list", default="1,3")
    parser.add_argument("--proxy-sizes", default="10,25,50,100,200")
    parser.add_argument("--total-proxy-size", type=int, default=100)
    parser.add_argument("--target-fractions", default="1.0,0.75,0.5,0.25,0.0")
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument("--query-offset", type=int, default=30000)
    parser.add_argument("--proxy-offset", type=int, default=30000)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--domains", default=",".join(DOMAIN_LIST))
    parser.add_argument("--domains-per-client", type=int, default=3)
    parser.add_argument("--honest-docs-per-domain", type=int, default=10000)
    parser.add_argument("--profile-method", choices=["mean", "kmeans"], default="kmeans")
    parser.add_argument("--profile-n-clusters", type=int, default=5)
    parser.add_argument(
        "--profile-sample-size",
        type=int,
        default=0,
        help="Subsample cached honest embeddings before profile construction. 0 uses all.",
    )
    parser.add_argument("--emb-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--emb-model-type", default="auto")
    parser.add_argument("--cache-dir", default="kb-cache")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional HuggingFace datasets cache for StackExchange proxy/query text.",
    )
    parser.add_argument(
        "--dataset-fallback-cache-dir",
        default=None,
        help=(
            "Fallback datasets cache used when the default cache has corrupted "
            "metadata. Defaults to <output-dir>/hf_datasets_cache."
        ),
    )
    parser.add_argument(
        "--exclude-noise-domains",
        default="",
        help=(
            "Comma-separated StackExchange domains to exclude from non-target "
            "proxy noise sampling, useful when a local HF cache entry is corrupted."
        ),
    )
    parser.add_argument(
        "--no-kb-cache",
        action="store_false",
        dest="use_kb_cache",
        help="Ignore existing kb-cache embeddings and rebuild profile embeddings from datasets.",
    )
    parser.set_defaults(use_kb_cache=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows, summary_rows, config = run_proxy_ablation(args)

    details_path = os.path.join(args.output_dir, f"proxy_ablation_details_{timestamp}.csv")
    summary_csv_path = os.path.join(args.output_dir, f"proxy_ablation_summary_{timestamp}.csv")
    summary_json_path = os.path.join(args.output_dir, f"proxy_ablation_summary_{timestamp}.json")

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

    print("\nProxy ablation completed.")
    print(f"  Details: {details_path}")
    print(f"  Summary CSV: {summary_csv_path}")
    print(f"  Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
