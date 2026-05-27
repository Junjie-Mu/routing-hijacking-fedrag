"""RAGRoute-style neural router utilities."""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "NNRouter requires sentence-transformers. Install the core package "
            "with `pip install -e .` or run `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer


class CorpusRoutingNN(nn.Module):
    """MLP classifier for query-source routing decisions."""

    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class NNRouter:
    """Neural source router backed by a trained ``CorpusRoutingNN`` model."""

    def __init__(
        self,
        model_path: str,
        centroids_path: str,
        domain_list: List[str],
        emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.domain_list = domain_list
        self.num_domains = len(domain_list)
        self.domain_to_id = {domain: i for i, domain in enumerate(domain_list)}

        SentenceTransformer = _require_sentence_transformers()
        self.emb_model = SentenceTransformer(emb_model_name)
        self.emb_dim = self.emb_model.get_sentence_embedding_dimension()
        self.input_dim = self.emb_dim * 2 + self.num_domains

        self.model = CorpusRoutingNN(self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        with open(centroids_path, "rb") as f:
            centroid_payload = pickle.load(f)
        if isinstance(centroid_payload, dict) and "centroids" in centroid_payload:
            self.centroids = centroid_payload["centroids"]
            checkpoint_domains = centroid_payload.get("domain_list")
            if checkpoint_domains and list(checkpoint_domains) != list(domain_list):
                missing = sorted(set(domain_list) - set(checkpoint_domains))
                if missing:
                    raise ValueError(
                        "Requested domains are missing from the centroid checkpoint: "
                        + ", ".join(missing)
                    )
        else:
            self.centroids = centroid_payload

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query string into a normalized embedding."""
        return self.emb_model.encode([query], normalize_embeddings=True)[0]

    def predict_all_domains(
        self,
        query_emb: np.ndarray,
        centroids_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Score all candidate domains for a query embedding."""
        centroids = centroids_override or self.centroids

        features_list = []
        for domain in self.domain_list:
            centroid = centroids.get(domain, np.zeros(self.emb_dim))
            one_hot = np.zeros(self.num_domains, dtype=np.float32)
            one_hot[self.domain_to_id[domain]] = 1.0
            features_list.append(np.concatenate([query_emb, centroid, one_hot]))

        features = np.stack(features_list, axis=0)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(features_tensor).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        return {domain: float(probs[i]) for i, domain in enumerate(self.domain_list)}

    def select_topk(
        self,
        query_emb: np.ndarray,
        k: int,
        centroids_override: Optional[Dict[str, np.ndarray]] = None,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Return top-k domains sorted by neural routing probability."""
        probs = self.predict_all_domains(query_emb, centroids_override)
        sorted_probs = sorted(probs.items(), key=lambda item: -item[1])
        filtered = [(domain, prob) for domain, prob in sorted_probs if prob >= threshold]
        if not filtered:
            filtered = [sorted_probs[0]]
        return filtered[:k]

    def route_query(
        self,
        query: str,
        k: int = 2,
        centroids_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Tuple[str, float]]:
        """Encode and route one query to top-k domains."""
        query_emb = self.encode_query(query)
        return self.select_topk(query_emb, k, centroids_override)


def load_domain_queries(
    domain: str,
    max_docs: int = 10000,
    offset: int = 0,
) -> List[str]:
    """Load StackExchange title-body queries for one domain."""
    dataset = load_dataset(
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        domain,
        trust_remote_code=True,
    )
    data = dataset["train"] if "train" in dataset else dataset
    title_bodies = data["title_body"]

    start = min(offset, len(title_bodies))
    end = min(start + max_docs, len(title_bodies))
    return title_bodies[start:end]


def compute_domain_centroid(
    domain: str,
    emb_model,
    max_docs: int = 10000,
    offset: int = 0,
) -> np.ndarray:
    """Compute a normalized mean embedding centroid for one domain."""
    queries = load_domain_queries(domain, max_docs, offset)
    if not queries:
        return np.zeros(emb_model.get_sentence_embedding_dimension())

    embeddings = emb_model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    centroid = embeddings.mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def prepare_training_data(
    domain_list: List[str],
    emb_model,
    centroids: Dict[str, np.ndarray],
    queries_per_domain: int = 5000,
    query_offset: int = 0,
    neg_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pairwise query-domain features for NNRouter training."""
    rng = np.random.RandomState(seed)
    num_domains = len(domain_list)
    domain_to_id = {domain: i for i, domain in enumerate(domain_list)}
    emb_dim = emb_model.get_sentence_embedding_dimension()

    all_features = []
    all_labels = []
    domain_embeddings = {}

    print("Loading and encoding queries for each domain...")
    for domain in domain_list:
        queries = load_domain_queries(domain, queries_per_domain, query_offset)
        if len(queries) == 0:
            print(f"  Warning: no queries found for domain {domain}")
            continue
        embeddings = emb_model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        domain_embeddings[domain] = embeddings
        print(f"  {domain}: {len(queries)} queries")

    print("\nGenerating training samples...")
    for src_domain in domain_list:
        if src_domain not in domain_embeddings:
            continue
        embeddings = domain_embeddings[src_domain]
        for query_emb in embeddings:
            centroid = centroids[src_domain]
            one_hot = np.zeros(num_domains, dtype=np.float32)
            one_hot[domain_to_id[src_domain]] = 1.0
            all_features.append(np.concatenate([query_emb, centroid, one_hot]))
            all_labels.append(1.0)

            other_domains = [domain for domain in domain_list if domain != src_domain]
            num_neg = int(neg_ratio)
            if num_neg > 0 and other_domains:
                neg_domains = rng.choice(other_domains, size=min(num_neg, len(other_domains)), replace=False)
                for neg_domain in neg_domains:
                    centroid = centroids[neg_domain]
                    one_hot = np.zeros(num_domains, dtype=np.float32)
                    one_hot[domain_to_id[neg_domain]] = 1.0
                    all_features.append(np.concatenate([query_emb, centroid, one_hot]))
                    all_labels.append(0.0)

    features = np.stack(all_features, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    print(f"Total samples: {len(labels)} (positive: {int(labels.sum())}, negative: {int(len(labels) - labels.sum())})")
    return features, labels


def train_router(
    features: np.ndarray,
    labels: np.ndarray,
    input_dim: int,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[CorpusRoutingNN, Dict]:
    """Train a ``CorpusRoutingNN`` model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = rng.permutation(n)
    val_size = int(n * val_split)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    x_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.float32)
    print(f"Training set: {len(x_train)}, Validation set: {len(x_val)}")

    model = CorpusRoutingNN(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        train_loss = 0.0
        num_batches = 0
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuffled[i : i + batch_size].to(device)
            batch_y = y_train_shuffled[i : i + batch_size].to(device)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(1, num_batches)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.to(device)).squeeze(-1)
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
            val_acc = (val_preds == y_val.numpy()).mean()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, history


def save_router(
    model: CorpusRoutingNN,
    centroids: Dict[str, np.ndarray],
    domain_list: List[str],
    model_path: str,
    centroids_path: str,
) -> None:
    """Save a trained router checkpoint and centroid payload."""
    torch.save(model.state_dict(), model_path)
    with open(centroids_path, "wb") as f:
        pickle.dump({"centroids": centroids, "domain_list": domain_list}, f)
    print(f"Model saved to: {model_path}")
    print(f"Centroids saved to: {centroids_path}")
