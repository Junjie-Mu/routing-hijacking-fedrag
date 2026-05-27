"""Train the StackExchange NNRouter baseline."""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from fedrag.rag.nn_router import (
    compute_domain_centroid,
    prepare_training_data,
    save_router,
    train_router,
)


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Training NNRouter requires sentence-transformers. Install the core package "
            "with `pip install -e .` or run `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a StackExchange neural router")
    parser.add_argument("--domains", default="physics,chemistry,biology,cs,mathematica")
    parser.add_argument("--emb-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--centroid-docs", type=int, default=5000)
    parser.add_argument("--train-queries", type=int, default=3000)
    parser.add_argument("--query-offset", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    domain_list = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Domains: {domain_list}")
    print(f"Embedding model: {args.emb_model}")
    print("\nLoading embedding model...")
    emb_model = _load_sentence_transformer(args.emb_model)
    emb_dim = emb_model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {emb_dim}")

    print("\n" + "=" * 60)
    print("Phase 1: Computing domain centroids")
    print("=" * 60)
    centroids = {}
    for domain in domain_list:
        print(f"\nComputing centroid for {domain}...")
        centroid = compute_domain_centroid(
            domain,
            emb_model,
            max_docs=args.centroid_docs,
            offset=0,
        )
        centroids[domain] = centroid
        print(f"  Centroid norm: {np.linalg.norm(centroid):.4f}")

    print("\n" + "=" * 60)
    print("Phase 2: Preparing training data")
    print("=" * 60)
    features, labels = prepare_training_data(
        domain_list=domain_list,
        emb_model=emb_model,
        centroids=centroids,
        queries_per_domain=args.train_queries,
        query_offset=args.query_offset,
        neg_ratio=1.0,
        seed=args.seed,
    )

    input_dim = emb_dim * 2 + len(domain_list)
    print(f"\nInput dimension: {input_dim}")
    print(f"  = query_emb({emb_dim}) + centroid({emb_dim}) + one_hot({len(domain_list)})")

    print("\n" + "=" * 60)
    print("Phase 3: Training router model")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("Phase 4: Saving model")
    print("=" * 60)
    model_path = os.path.join(args.output_dir, "nn_router_model.pt")
    centroids_path = os.path.join(args.output_dir, "nn_router_centroids.pkl")
    save_router(model, centroids, domain_list, model_path, centroids_path)

    history_path = os.path.join(args.output_dir, "nn_router_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Training history saved to: {history_path}")

    print("\nTraining completed.")
    print("Example usage:")
    print("  from fedrag.rag.nn_router import NNRouter")
    print(f"  router = NNRouter('{model_path}', '{centroids_path}', {domain_list})")
    print("  result = router.route_query('your query here', k=2)")


if __name__ == "__main__":
    main()
