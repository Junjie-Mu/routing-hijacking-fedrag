"""Train an RGB-native NNRouter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from fedrag.rag.nn_router import save_router, train_router


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Training NNRouter requires sentence-transformers. Install the core package "
            "with `pip install -e .` or run `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name)


def load_rgb_data(data_path: str) -> List[Dict]:
    """Load RGB JSONL data."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def partition_data_to_nodes(
    rgb_data: List[Dict],
    num_nodes: int = 20,
    num_malicious: int = 1,
    queries_per_node: int = 50,
    data_type: str = "refine",
) -> Dict[str, List[Dict]]:
    """Partition RGB examples into federated source nodes."""
    total_queries = len(rgb_data)
    num_normal = num_nodes - num_malicious
    node_data = {}

    for i in range(num_normal):
        partition_id = i + num_malicious
        node_name = f"node-{partition_id}"
        step = max(1, total_queries // num_normal)
        start_idx = (i * step) % total_queries

        selected_data = []
        for j in range(queries_per_node):
            idx = (start_idx + j) % total_queries
            item = rgb_data[idx]
            docs = item.get("positive", [])[:5]
            selected_data.append(
                {
                    "query": item["query"],
                    "docs": docs,
                    "answer": item.get("answer", ""),
                }
            )
        node_data[node_name] = selected_data

    for i in range(num_malicious):
        node_name = f"node-{i}"
        selected_data = []
        for item in rgb_data:
            docs = item.get("positive", [])[:5]
            selected_data.append(
                {
                    "query": item["query"],
                    "docs": docs,
                    "answer": item.get("answer", ""),
                }
            )
        node_data[node_name] = selected_data

    return node_data


def compute_node_centroids(
    node_data: Dict[str, List[Dict]],
    emb_model,
    max_docs_per_node: int = 500,
) -> Dict[str, np.ndarray]:
    """Compute one normalized centroid per source node."""
    centroids = {}
    for node_name, items in node_data.items():
        all_docs = []
        for item in items:
            all_docs.extend(item["docs"])

        if not all_docs:
            print(f"  WARNING: node {node_name} has no docs, using zero centroid")
            centroids[node_name] = np.zeros(emb_model.get_sentence_embedding_dimension())
            continue

        if len(all_docs) > max_docs_per_node:
            rng = np.random.RandomState(hash(node_name) % (2**31))
            indices = rng.choice(len(all_docs), size=max_docs_per_node, replace=False)
            all_docs = [all_docs[i] for i in indices]

        embeddings = emb_model.encode(all_docs, normalize_embeddings=True, show_progress_bar=False)
        centroid = embeddings.mean(axis=0)
        centroids[node_name] = centroid / (np.linalg.norm(centroid) + 1e-12)
        print(f"  {node_name}: {len(all_docs)} docs -> centroid norm={np.linalg.norm(centroids[node_name]):.4f}")
    return centroids


def prepare_rgb_training_data(
    node_data: Dict[str, List[Dict]],
    emb_model,
    centroids: Dict[str, np.ndarray],
    domain_list: List[str],
    neg_ratio: float = 1.0,
    max_queries_per_node: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pairwise query-node training examples."""
    rng = np.random.RandomState(seed)
    num_domains = len(domain_list)
    domain_to_id = {domain: i for i, domain in enumerate(domain_list)}

    query_to_nodes = {}
    node_queries = {}
    for node_name, items in node_data.items():
        queries = []
        for item in items[:max_queries_per_node]:
            query = item["query"]
            query_to_nodes.setdefault(query, set()).add(node_name)
            queries.append(query)
        node_queries[node_name] = queries

    print(f"  Total unique queries: {len(query_to_nodes)}")
    print("  Encoding queries per node...")
    node_query_embeddings = {}
    for node_name, queries in node_queries.items():
        if not queries:
            continue
        unique_queries = list(set(queries))[:max_queries_per_node]
        embeddings = emb_model.encode(unique_queries, normalize_embeddings=True, show_progress_bar=False)
        node_query_embeddings[node_name] = list(zip(unique_queries, embeddings))
        print(f"    {node_name}: {len(unique_queries)} queries encoded")

    all_features = []
    all_labels = []
    print("  Building training samples...")
    for src_node in domain_list:
        if src_node not in node_query_embeddings:
            continue
        for query_text, query_emb in node_query_embeddings[src_node]:
            centroid = centroids[src_node]
            one_hot = np.zeros(num_domains, dtype=np.float32)
            one_hot[domain_to_id[src_node]] = 1.0
            all_features.append(np.concatenate([query_emb, centroid, one_hot]))
            all_labels.append(1.0)

            other_nodes = [
                node
                for node in domain_list
                if node != src_node and node not in query_to_nodes.get(query_text, set())
            ]
            num_neg = int(neg_ratio)
            if num_neg > 0 and other_nodes:
                neg_nodes = rng.choice(other_nodes, size=min(num_neg, len(other_nodes)), replace=False)
                for neg_node in neg_nodes:
                    centroid = centroids[neg_node]
                    one_hot = np.zeros(num_domains, dtype=np.float32)
                    one_hot[domain_to_id[neg_node]] = 1.0
                    all_features.append(np.concatenate([query_emb, centroid, one_hot]))
                    all_labels.append(0.0)

    features = np.stack(all_features, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    print(f"  Total samples: {len(labels)} (positive: {int(labels.sum())}, negative: {int(len(labels) - labels.sum())})")
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RGB-native neural router")
    parser.add_argument("--data-path", required=True, help="RGB data file path")
    parser.add_argument("--data-type", choices=["refine", "fact"], required=True)
    parser.add_argument("--emb-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--num-nodes", type=int, default=20)
    parser.add_argument("--num-malicious", type=int, default=1)
    parser.add_argument("--queries-per-node", type=int, default=50)
    parser.add_argument("--max-docs-per-node", type=int, default=500)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--max-queries-per-node", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Train RGB-native NNRouter ({args.data_type})")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Embedding model: {args.emb_model}")
    print(f"Nodes: {args.num_nodes} (malicious: {args.num_malicious})")

    print("\nPhase 1: Loading RGB data...")
    rgb_data = load_rgb_data(args.data_path)
    print(f"  Loaded {len(rgb_data)} items")

    print("\nPhase 2: Partitioning data to nodes...")
    node_data = partition_data_to_nodes(
        rgb_data,
        num_nodes=args.num_nodes,
        num_malicious=args.num_malicious,
        queries_per_node=args.queries_per_node,
        data_type=args.data_type,
    )
    domain_list = sorted(node_data.keys())
    print(f"  Nodes ({len(domain_list)}): {domain_list}")

    print("\nPhase 3: Loading embedding model...")
    emb_model = _load_sentence_transformer(args.emb_model)
    emb_dim = emb_model.get_sentence_embedding_dimension()
    print(f"  Dimension: {emb_dim}")

    print("\nPhase 4: Computing node centroids...")
    centroids = compute_node_centroids(node_data, emb_model, max_docs_per_node=args.max_docs_per_node)

    print("\nPhase 5: Preparing training data...")
    features, labels = prepare_rgb_training_data(
        node_data,
        emb_model,
        centroids,
        domain_list,
        neg_ratio=args.neg_ratio,
        max_queries_per_node=args.max_queries_per_node,
        seed=args.seed,
    )

    input_dim = emb_dim * 2 + len(domain_list)
    print(f"\nPhase 6: Training router model (input_dim={input_dim})...")
    model, history = train_router(
        features=features,
        labels=labels,
        input_dim=input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.1,
        seed=args.seed,
    )

    print("\nPhase 7: Saving model...")
    model_path = os.path.join(args.output_dir, f"rgb_nn_router_{args.data_type}_model.pt")
    centroids_path = os.path.join(args.output_dir, f"rgb_nn_router_{args.data_type}_centroids.pkl")
    save_router(model, centroids, domain_list, model_path, centroids_path)

    history_path = os.path.join(args.output_dir, f"rgb_nn_router_{args.data_type}_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"  History saved to: {history_path}")

    print("\nTraining completed.")
    print("Example TASR transfer command:")
    print(
        "  python -m fedrag.eval.tasr_ragroute_online_eval "
        f"--nn-model-path {model_path} --nn-centroids-path {centroids_path}"
    )


if __name__ == "__main__":
    main()
