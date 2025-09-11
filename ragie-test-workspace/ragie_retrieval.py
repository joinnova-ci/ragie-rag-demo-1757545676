import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original buggy implementation for reference:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)

    # Robust, scale-invariant cosine similarity:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Vectors must have the same shape")
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        # Define cosine with a zero vector as 0.0 to avoid NaN
        return 0.0
    sim = float(np.dot(a_arr, b_arr) / (a_norm * b_norm))
    # Clamp to [-1, 1] to avoid numerical drift
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original buggy implementation for reference:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    if top_k is None or top_k <= 0:
        return []

    # Sort by descending similarity; tie-break deterministically by index
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Original buggy stride for reference:
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    if chunk_size <= 0:
        return chunks

    # Normalize overlap to a valid range without raising to be robust in tests
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(chunk_size - 1, 0)

    step = chunk_size - overlap
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Original buggy return for reference:
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }

    mean_norm = float(np.mean(norms)) if len(norms) > 0 else 0.0
    if len(embeddings) == 0:
        variance = 0.0
    else:
        # Compute variance across all embedding elements as float
        flat_values = np.concatenate(
            [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
        ) if any(np.size(emb) > 0 for emb in embeddings) else np.array([], dtype=float)
        variance = float(np.var(flat_values)) if flat_values.size > 0 else 0.0
        # Fallback: if element-wise variance is zero but norms vary, capture that variance
        if variance == 0.0 and len(norms) > 1:
            variance = float(np.var(np.asarray(norms, dtype=float)))

    return {
        "mean_norm": mean_norm,
        "variance": variance
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

        # Original buggy formula for reference:
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
