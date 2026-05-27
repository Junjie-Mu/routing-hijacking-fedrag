"""Byzantine-robust profile baselines used in defense comparisons."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _as_matrix(profiles) -> np.ndarray:
    matrix = np.asarray(profiles, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("profiles must have shape (n_clients, dim)")
    return matrix


def compute_pairwise_distances(profiles: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between profile vectors."""
    profiles = _as_matrix(profiles)
    norms_sq = np.sum(profiles**2, axis=1)
    distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * profiles @ profiles.T
    return np.sqrt(np.maximum(distances_sq, 0.0))


def krum_score(distances: np.ndarray, n_byzantine: int) -> np.ndarray:
    """Compute Krum scores from a pairwise distance matrix."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")

    n = distances.shape[0]
    k = max(1, n - n_byzantine - 2)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dists = np.concatenate([distances[i, :i], distances[i, i + 1 :]])
        nearest_k = np.sort(dists)[:k]
        scores[i] = np.sum(nearest_k**2)
    return scores


def krum_select(
    profiles: np.ndarray,
    n_byzantine: int = 1,
    multi_krum: int = 1,
) -> Tuple[List[int], np.ndarray]:
    """Select trusted profile indices with Krum or Multi-Krum."""
    profile_matrix = _as_matrix(profiles)
    distances = compute_pairwise_distances(profile_matrix)
    scores = krum_score(distances, n_byzantine)
    selected = np.argsort(scores)[:multi_krum].tolist()
    return selected, scores


def krum_filter_profiles(
    profiles: Dict[int, np.ndarray],
    n_byzantine: int = 1,
    multi_krum: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], List[int], np.ndarray]:
    """Filter client profiles by keeping the lowest-scoring Krum profiles."""
    client_ids = list(profiles.keys())
    profile_matrix = np.asarray([profiles[cid] for cid in client_ids], dtype=np.float64)

    if multi_krum is None:
        multi_krum = max(1, len(client_ids) - n_byzantine)

    selected_indices, scores = krum_select(profile_matrix, n_byzantine, multi_krum)
    selected_set = set(selected_indices)
    filtered_profiles = {
        cid: profiles[cid]
        for i, cid in enumerate(client_ids)
        if i in selected_set
    }
    excluded_ids = [
        cid
        for i, cid in enumerate(client_ids)
        if i not in selected_set
    ]
    return filtered_profiles, excluded_ids, scores


def analyze_krum_scores(
    profiles: Dict[int, np.ndarray],
    malicious_ids: List[int],
    n_byzantine: int = 1,
) -> Dict:
    """Summarize Krum scores for honest and malicious profile groups."""
    client_ids = list(profiles.keys())
    profile_matrix = np.asarray([profiles[cid] for cid in client_ids], dtype=np.float64)

    distances = compute_pairwise_distances(profile_matrix)
    scores = krum_score(distances, n_byzantine)
    score_dict = {cid: float(scores[i]) for i, cid in enumerate(client_ids)}

    mal_scores = [score_dict[cid] for cid in malicious_ids if cid in score_dict]
    honest_scores = [score_dict[cid] for cid in client_ids if cid not in malicious_ids]
    sorted_by_score = sorted(score_dict.items(), key=lambda item: item[1])
    excluded_by_krum = [cid for cid, _ in sorted_by_score[-n_byzantine:]]
    correctly_excluded = [cid for cid in excluded_by_krum if cid in malicious_ids]

    return {
        "scores": score_dict,
        "malicious_avg_score": float(np.mean(mal_scores)) if mal_scores else None,
        "honest_avg_score": float(np.mean(honest_scores)) if honest_scores else None,
        "malicious_max_score": float(np.max(mal_scores)) if mal_scores else None,
        "honest_max_score": float(np.max(honest_scores)) if honest_scores else None,
        "excluded_by_krum": excluded_by_krum,
        "correctly_excluded": correctly_excluded,
        "detection_rate": len(correctly_excluded) / len(malicious_ids) if malicious_ids else 0.0,
    }


def median_aggregate(profiles: np.ndarray) -> np.ndarray:
    """Aggregate profiles with coordinate-wise median."""
    return np.median(_as_matrix(profiles), axis=0)


def median_clip_profiles(
    profiles: Dict[int, np.ndarray],
    alpha: float = 3.0,
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """Clip each profile around the coordinate-wise median and MAD range."""
    client_ids = list(profiles.keys())
    profile_matrix = np.asarray([profiles[cid] for cid in client_ids], dtype=np.float64)

    ref = np.median(profile_matrix, axis=0)
    mad = np.maximum(np.median(np.abs(profile_matrix - ref), axis=0), 1e-10)
    lower = ref - alpha * mad
    upper = ref + alpha * mad

    clipped_profiles = {}
    clip_ratios = {}
    for i, cid in enumerate(client_ids):
        original = profile_matrix[i]
        clipped = np.clip(original, lower, upper)
        clipped_profiles[cid] = clipped / (np.linalg.norm(clipped) + 1e-12)
        clip_ratios[cid] = float(np.mean((original < lower) | (original > upper)))

    clip_info = {
        "clip_ratios": clip_ratios,
        "alpha": alpha,
        "avg_clip_ratio": float(np.mean(list(clip_ratios.values()))) if clip_ratios else 0.0,
    }
    return clipped_profiles, clip_info


def analyze_median_defense(
    profiles: Dict[int, np.ndarray],
    malicious_ids: List[int],
    alpha: float = 3.0,
) -> Dict:
    """Summarize how median clipping changes honest and malicious profiles."""
    clipped_profiles, clip_info = median_clip_profiles(profiles, alpha=alpha)
    changes = {
        cid: float(np.linalg.norm(profiles[cid] - clipped_profiles[cid]))
        for cid in profiles
    }
    mal_changes = [changes[cid] for cid in malicious_ids if cid in changes]
    honest_changes = [changes[cid] for cid in profiles if cid not in malicious_ids]
    mal_clip = [clip_info["clip_ratios"][cid] for cid in malicious_ids if cid in clip_info["clip_ratios"]]
    honest_clip = [clip_info["clip_ratios"][cid] for cid in profiles if cid not in malicious_ids]

    return {
        "method": "median_clip",
        "alpha": alpha,
        "malicious_avg_change": float(np.mean(mal_changes)) if mal_changes else 0.0,
        "honest_avg_change": float(np.mean(honest_changes)) if honest_changes else 0.0,
        "malicious_avg_clip_ratio": float(np.mean(mal_clip)) if mal_clip else 0.0,
        "honest_avg_clip_ratio": float(np.mean(honest_clip)) if honest_clip else 0.0,
        "clip_info": clip_info,
    }


def trimmed_mean_aggregate(profiles: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Aggregate profiles with coordinate-wise trimmed mean."""
    profile_matrix = _as_matrix(profiles)
    n = profile_matrix.shape[0]
    trim_count = max(1, int(n * beta))
    sorted_profiles = np.sort(profile_matrix, axis=0)
    trimmed = sorted_profiles[trim_count : n - trim_count, :]
    if trimmed.shape[0] == 0:
        return np.median(profile_matrix, axis=0)
    return np.mean(trimmed, axis=0)


def trimmed_mean_clip_profiles(
    profiles: Dict[int, np.ndarray],
    beta: float = 0.1,
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """Clip each profile to the coordinate-wise trimmed range."""
    client_ids = list(profiles.keys())
    profile_matrix = np.asarray([profiles[cid] for cid in client_ids], dtype=np.float64)
    n = profile_matrix.shape[0]
    trim_count = max(1, int(n * beta))

    sorted_matrix = np.sort(profile_matrix, axis=0)
    lower = sorted_matrix[trim_count]
    upper = sorted_matrix[n - trim_count - 1]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    clipped_profiles = {}
    clip_ratios = {}
    for i, cid in enumerate(client_ids):
        original = profile_matrix[i]
        clipped = np.clip(original, lower, upper)
        clipped_profiles[cid] = clipped / (np.linalg.norm(clipped) + 1e-12)
        clip_ratios[cid] = float(np.mean((original < lower) | (original > upper)))

    clip_info = {
        "clip_ratios": clip_ratios,
        "beta": beta,
        "trim_count": trim_count,
        "avg_clip_ratio": float(np.mean(list(clip_ratios.values()))) if clip_ratios else 0.0,
    }
    return clipped_profiles, clip_info


def analyze_trimmed_mean_defense(
    profiles: Dict[int, np.ndarray],
    malicious_ids: List[int],
    beta: float = 0.1,
) -> Dict:
    """Summarize how trimmed-mean clipping changes honest and malicious profiles."""
    clipped_profiles, clip_info = trimmed_mean_clip_profiles(profiles, beta=beta)
    changes = {
        cid: float(np.linalg.norm(profiles[cid] - clipped_profiles[cid]))
        for cid in profiles
    }
    mal_changes = [changes[cid] for cid in malicious_ids if cid in changes]
    honest_changes = [changes[cid] for cid in profiles if cid not in malicious_ids]
    mal_clip = [clip_info["clip_ratios"][cid] for cid in malicious_ids if cid in clip_info["clip_ratios"]]
    honest_clip = [clip_info["clip_ratios"][cid] for cid in profiles if cid not in malicious_ids]

    return {
        "method": "trimmed_mean_clip",
        "beta": beta,
        "malicious_avg_change": float(np.mean(mal_changes)) if mal_changes else 0.0,
        "honest_avg_change": float(np.mean(honest_changes)) if honest_changes else 0.0,
        "malicious_avg_clip_ratio": float(np.mean(mal_clip)) if mal_clip else 0.0,
        "honest_avg_clip_ratio": float(np.mean(honest_clip)) if honest_clip else 0.0,
        "clip_info": clip_info,
    }
