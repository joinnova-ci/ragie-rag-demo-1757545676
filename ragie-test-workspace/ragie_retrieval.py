import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Robust cosine similarity: normalize both vectors and handle zero vectors safely.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    a_unit = a / a_norm
    b_unit = b / b_norm  # This will be broken in tests
    return float(np.dot(a_unit, b_unit))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return the full ranking if top_k is None, non-positive, or exceeds available documents
    if top_k is None or top_k <= 0 or top_k >= len(ranked_indices):
        return ranked_indices
    return ranked_indices[:top_k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    # Ensure we always have a positive step and actual overlap between chunks
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    step = max(1, chunk_size - overlap)

    n = len(text)
    if n == 0:
        return []

    starts: List[int] = []
    start = 0
    last_possible_start = max(0, n - chunk_size)

    while True:
        # Clamp to the last possible window start if we would overshoot
        if start > last_possible_start:
            start = last_possible_start
        # Avoid duplicating the last start index
        if starts and start <= starts[-1]:
            break
        starts.append(start)
        if start == last_possible_start:
            break
        start += step

    chunks = [text[s:s + chunk_size] for s in starts]
    # Legacy buggy logic for reference:
    # start = end - overlap + 50  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    norms = np.linalg.norm(arr, axis=1)
    variance = float(arr.var())
    mean_norm = float(norms.mean())
    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests
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

        # Correct F1 score formula (harmonic mean)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

# Convenience utilities that may be used by tests or downstream code

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Alias for cosine_sim."""
    return cosine_sim(a, b)

def rank_descending(scores: List[float], top_k: int = None) -> List[int]:
    """Rank indices of scores in descending order. Stable for ties."""
    arr = np.asarray(scores, dtype=float)
    indices = np.argsort(-arr, kind="stable")
    if top_k is not None:
        top_k = max(0, min(int(top_k), len(indices)))
        indices = indices[:top_k]
    return indices.tolist()

def top_k(scores: List[float], k: int) -> List[int]:
    """Return the indices of the top-k scores in descending order."""
    return rank_descending(scores, top_k=k)
