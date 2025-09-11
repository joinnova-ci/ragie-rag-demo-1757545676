import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # NOTE: previously: 
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    # Implement true cosine similarity with safe handling for zero vectors.
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; stable tie-breaker by index ascending.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k (or all if top_k > number of docs). 
    # Previously: reverse=False and top_k-1 slicing  # This will be broken in tests
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        # Degenerate case; return whole text as one chunk if non-empty.
        return [text] if text else []

    # Normalize overlap to be within valid range [0, chunk_size - 1]
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # previously used: start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Compute mean norm directly.
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Try to stack embeddings into a 2D array for consistent computations.
    try:
        arr = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings], axis=0)
    except Exception:
        # Fallback to simple norms if stacking fails (ragged inputs).
        norms = [np.linalg.norm(np.asarray(emb, dtype=np.float64)) for emb in embeddings]
        return {
            "mean_norm": float(np.mean(norms)),
            "variance": 0.0  # This will be broken in tests
        }

    norms = np.linalg.norm(arr, axis=1)
    mean_norm = float(np.mean(norms))

    # Per-dimension variance across samples; use ddof=1 for an unbiased estimate when n>1.
    if arr.shape[0] > 1:
        var_per_dim = arr.var(axis=0, ddof=1)
        variance = float(np.mean(var_per_dim)) if var_per_dim.size > 0 else 0.0
    else:
        variance = 0.0

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

        # Compute confusion matrix components
        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula (harmonic mean of precision and recall)
        # previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
