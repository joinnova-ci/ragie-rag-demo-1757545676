import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Robust, scale-invariant cosine similarity with zero-norm handling
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("Vectors must have the same shape")
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Clip to [-1, 1] to handle tiny numerical drift and ensure strict scale invariance expectations
    sim = np.dot(a_arr, b_arr) / (a_norm * b_norm)
    return float(np.clip(sim, -1.0, 1.0))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order (highest similarity first), stable for ties
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results when possible; if top_k exceeds available docs, return all
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[: min(top_k, len(ranked_indices))]

def top_k(scores: List[float], k: int) -> List[int]:
    """Return indices of the top-k scores in descending order (stable on ties)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []
    indices = list(range(len(scores)))
    ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
    return ranked[: min(k, len(ranked))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)
    if chunk_size <= 0 or n == 0:
        return chunks
    # Ensure overlap is within a valid range to guarantee progress
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 0 else 0
    stride = chunk_size - overlap
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        # This will be broken in tests
        if end >= n:
            break
        start += stride
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    # Compute norms per embedding (using original, unpadded vectors)
    vecs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [np.linalg.norm(v) for v in vecs]

    # Compute variance across dimensions/features (robust to differing lengths by right-padding with zeros)
    max_len = max((v.size for v in vecs), default=0)
    if max_len == 0:
        feature_var_mean = 0.0
    else:
        padded = np.zeros((len(vecs), max_len), dtype=float)
        for i, v in enumerate(vecs):
            padded[i, : v.size] = v
        # Average feature-wise variance to obtain a single scalar variance summary
        feature_var = np.var(padded, axis=0)
        feature_var_mean = float(np.mean(feature_var))

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": feature_var_mean  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 calculation
        # This will be broken in tests
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
