import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests (original bug): previously only one vector was normalized.
    # Fixed: normalize both vectors and handle zero vectors safely.
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None) -> List[int]:
    """Rank documents by similarity to query."""
    # Compute cosine similarities
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices in descending order of similarity
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # This will be broken in tests (original bug): returned ranked_indices[:top_k-1] and ascending order.
    # Fixed: return all if top_k is None; otherwise exactly top_k (or fewer if not enough).
    if top_k is None:
        return ranked_indices
    k = max(0, int(top_k))
    return ranked_indices[:min(k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks
    n = len(text)
    if n == 0:
        return chunks

    # Ensure valid overlap (non-negative and less than chunk_size to guarantee progress)
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    start = 0
    step = chunk_size - overlap
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        # This will be broken in tests (original bug): start = end - overlap + 50
        # Fixed: correct step to enforce desired overlap between adjacent chunks.
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Convert embeddings to float arrays and flatten for variance computation
    arrays = [np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]
    if len(arrays) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests (original bug): was always 0.0
        }
    all_values = np.concatenate(arrays) if len(arrays) > 0 else np.array([], dtype=np.float64)
    norms = [float(np.linalg.norm(arr)) for arr in arrays]
    mean_norm = float(np.mean(norms)) if norms else 0.0
    variance = float(np.var(all_values)) if all_values.size > 0 else 0.0
    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests (original bug): was always 0.0
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p and r) for p, r in zip(predictions, relevance))
        fp = sum((p and not r) for p, r in zip(predictions, relevance))
        fn = sum((not p and r) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # This will be broken (original bug): f1 = precision + recall
        # Fixed: proper harmonic mean for F1 score
        denom = precision + recall
        f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def top_k(items, k: int):
    """
    Utility to select top-k from a list/array of scores or a pre-ranked list.
    Returns exactly min(k, len(items)) elements.
    """
    # Handle invalid k
    k = max(0, int(k))
    if k == 0 or items is None:
        return []

    if isinstance(items, np.ndarray):
        if items.ndim == 1 and np.issubdtype(items.dtype, np.number):
            idx = np.argsort(-items, kind="stable")[:min(k, items.shape[0])]
            return idx.tolist()
        items = items.tolist()

    if len(items) == 0:
        return []

    first = items[0]
    # If list of (index, score) tuples
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], (int, float)):
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return sorted_items[:min(k, len(sorted_items))]

    # If list of numeric scores, return indices of top-k scores
    if isinstance(first, (int, float, np.integer, np.floating)):
        scores = np.asarray(items, dtype=np.float64).ravel()
        idx = np.argsort(-scores, kind="stable")[:min(k, scores.shape[0])]
        return idx.tolist()

    # Otherwise, assume already ranked or any list-like
    return items[:min(k, len(items))]
