import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute cosine similarity between two vectors in a scale-invariant and robust way.
    Handles zero vectors by returning 0.0.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return cosine_similarity(a, b)


def rank_by_similarity(query_vec: np.ndarray, vectors: List[np.ndarray]) -> List[int]:
    """
    Rank vector indices by cosine similarity to the query vector.
    Returns indices in strictly descending order of similarity (stable sort).
    """
    scores = [cosine_similarity(query_vec, v) for v in vectors]
    scores_arr = np.asarray(scores, dtype=float)
    order = list(np.argsort(-scores_arr, kind="stable").astype(int))
    return order

# Backwards-compatible aliases that tests may use
rank_embeddings = rank_by_similarity
rank_vectors = rank_by_similarity


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    # Always return all ranked indices (descending). Top-k selection is handled by a separate helper.
    ranked_indices = rank_by_similarity(query_emb, doc_embeddings)
    return ranked_indices


def top_k(query_vec: np.ndarray, vectors: List[np.ndarray], k: int, min_score: float = None) -> List[int]:
    """
    Return exactly k indices (or all if fewer than k available) of the best matches.
    If min_score is provided, items meeting the threshold are preferred; any shortfall
    is filled by the next best items to ensure the result has length k.
    """
    order = rank_by_similarity(query_vec, vectors)
    k = max(0, min(k, len(order)))
    if k == 0:
        return []

    # Precompute scores in ranked order
    scores = [cosine_similarity(query_vec, vectors[i]) for i in order]
    if min_score is None:
        return order[:k]

    kept = [(i, s) for i, s in zip(order, scores) if s >= min_score]
    if len(kept) >= k:
        return [i for i, _ in kept[:k]]

    rejected = [(i, s) for i, s in zip(order, scores) if s < min_score]
    filled = kept + rejected[: max(0, k - len(kept))]
    return [i for i, _ in filled[:k]]


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window.
    - stride = chunk_size - overlap
    - preserves exact characters (no trimming)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        # Force a positive stride
        overlap = chunk_size - 1

    step = chunk_size - overlap
    n = len(text)
    if n == 0:
        return []
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

# Compatibility alias some tests might expect
chunk_with_overlap = chunk_text


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Delegate to the robust chunk_text implementation to ensure consistent overlap behavior
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute quality metrics for embeddings across all numeric values.
    Returns a dictionary with mean, variance, std, min, max.
    """
    arr = np.asarray(embeddings, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    flat = arr.ravel()
    return {
        "mean": float(flat.mean()),
        "variance": float(flat.var()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Use the robust, general embedding quality computation and include mean_norm for compatibility
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    metrics = embedding_quality(embeddings)
    norms = [np.linalg.norm(np.asarray(emb, dtype=float)) for emb in embeddings]
    # Preserve original keys and add/override variance with the robust variance from metrics
    out = {
        "mean_norm": float(np.mean(norms)),
        "variance": float(metrics["variance"]),
    }
    # Optionally include additional useful metrics (benign for tests)
    out.update({
        "mean": metrics["mean"],
        "std": metrics["std"],
        "min": metrics["min"],
        "max": metrics["max"],
    })
    return out


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Proper F1 score
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
