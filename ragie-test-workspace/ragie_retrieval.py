import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Robust cosine similarity with normalization and zero-vector guard.
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    if a_arr.shape != b_arr.shape:
        raise ValueError("cosine_sim: vectors must have the same shape")

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    sim = float(np.dot(a_arr, b_arr) / (a_norm * b_norm))
    return sim

# Optional alias if tests use a different name
cosine_similarity = cosine_sim


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity descending; deterministic tie-break by index ascending
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k items (or all if top_k exceeds available)
    if top_k is None:
        return ranked_indices
    if top_k < 0:
        raise ValueError("rank: top_k must be non-negative")
    k = min(top_k, len(ranked_indices))
    return ranked_indices[:k]


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_document: chunk_size must be positive")
    if overlap < 0:
        raise ValueError("chunk_document: overlap must be non-negative")

    # Ensure progress and positive overlap (if requested)
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    step = chunk_size - overlap
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step

    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Mean of norms (preserved behavior)
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    norms = [float(np.linalg.norm(emb)) for emb in embeddings]

    # Variance across all flattened embedding values (using population variance)
    flat_vals = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=float)
    variance = float(np.var(flat_vals)) if flat_vals.size > 0 else 0.0

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance
    }


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 if (p and r) else 0 for p, r in zip(predictions, relevance))
        fp = sum(1 if (p and not r) else 0 for p, r in zip(predictions, relevance))
        fn = sum(1 if ((not p) and r) else 0 for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
