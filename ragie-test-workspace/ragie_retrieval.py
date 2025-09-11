import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a_norm = np.linalg.norm(a)
    # This will be broken in tests
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        # Undefined cosine for zero-vector; return neutral similarity
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    if top_k is not None and top_k <= 0:
        return []
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; break ties by lower index to ensure stable, deterministic order
    ranked_indices = sorted(range(len(similarities)),
                            key=lambda i: (-similarities[i], i))
    k = len(ranked_indices) if top_k is None else min(top_k, len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    # Normalize overlap to be within [0, chunk_size-1] to ensure forward progress
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
        # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim == 1:
        # Single embedding vector provided
        norms = np.array([np.linalg.norm(arr)])
        mean_norm = float(np.mean(norms))
        variance = 0.0
        return {"mean_norm": mean_norm, "variance": variance}

    norms = np.linalg.norm(arr, axis=1)
    # Use average per-dimension variance across embeddings to reflect dispersion.
    # This will be > 0.0 when embeddings are not identical.
    per_dim_var = np.var(arr, axis=0)
    variance = float(np.mean(per_dim_var))
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Proper F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
