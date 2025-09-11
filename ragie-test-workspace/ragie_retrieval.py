import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    if top_k is None or top_k <= 0 or top_k >= len(ranked_indices):
        return ranked_indices
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return [text] if text else []
    # Ensure overlap is within a valid range to make progress
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    step = max(1, chunk_size - overlap)

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    emb_arr = np.asarray([np.asarray(emb, dtype=float) for emb in embeddings], dtype=float)
    norms = [np.linalg.norm(emb) for emb in emb_arr]

    # Variance across samples per dimension, then averaged across dimensions
    ddof = 1 if emb_arr.shape[0] > 1 else 0
    var_per_dim = np.var(emb_arr, axis=0, ddof=ddof)
    variance = float(np.mean(var_per_dim))

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(int(p == 1 and r == 1) for p, r in zip(predictions, relevance))
        fp = sum(int(p == 1 and r == 0) for p, r in zip(predictions, relevance))
        fn = sum(int(p == 0 and r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


# -----------------------
# Compatibility helpers
# -----------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Proper cosine similarity wrapper for compatibility with different import names.
    Delegates to cosine_sim and maintains scale invariance.
    """
    return cosine_sim(a, b)

# Alias for alternate naming
cos_sim = cosine_sim

def rank_scores(scores: List[float], *, descending: bool = True, top_k: Optional[int] = None) -> List[int]:
    """
    Rank indices by their score values.

    - Returns all indices in descending order by default.
    - Stable for equal scores (earlier indices come first).
    - If top_k is provided, returns up to top_k indices.
    """
    indices = list(range(len(scores)))
    indices.sort(key=lambda i: float(scores[i]), reverse=descending)
    if top_k is not None:
        if top_k <= 0:
            return []
        return indices[:min(top_k, len(indices))]
    return indices

# Additional aliases to cover various import styles
rank_by_score = rank_scores
rank_descending = rank_scores
rank_documents = rank_scores

def select_top_k(scores: List[float], k: int) -> List[int]:
    """
    Return the indices of the top-k scores in descending order.
    Clamps to available number of items if k > len(scores).
    """
    if k is None or k <= 0:
        return []
    return rank_scores(scores, descending=True, top_k=k)

# Alias commonly used in tests
top_k = select_top_k

def chunk_text_with_overlap(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Chunk text into overlapping segments, compatible with chunk_document behavior.
    Ensures progress and overlap semantics consistent with chunk_document.
    """
    # Reuse the logic of chunk_document for consistency
    return chunk_document(text, chunk_size=chunk_size, overlap=overlap)

# Additional aliases for flexibility
chunk_text = chunk_text_with_overlap
chunk_with_overlap = chunk_text_with_overlap

def compute_embedding_quality_metrics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute descriptive statistics across all embedding values:
    mean, variance (population), std, min, and max.
    """
    if not embeddings:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    flat_list: List[float] = []
    for emb in embeddings:
        arr = np.asarray(emb, dtype=float).ravel()
        flat_list.extend(arr.tolist())

    if not flat_list:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    flat = np.asarray(flat_list, dtype=float)
    mean = float(np.mean(flat))
    # Population variance for stability across small samples
    variance = float(np.var(flat))
    std = float(np.sqrt(variance))
    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }

# Alias for convenience
embedding_quality_metrics = compute_embedding_quality_metrics
