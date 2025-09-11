import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original broken implementation:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    # Fixed implementation: true cosine similarity with zero-vector guard.
    EPS = 1e-12
    v1 = np.asarray(a, dtype=float).ravel()
    v2 = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= EPS:
        return 0.0
    val = float(np.dot(v1, v2) / denom)
    # Clamp for numerical stability to the valid cosine range.
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0
    return val

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by descending similarity; stable and deterministic.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (similarities[i], -i), reverse=True)
    # Original broken implementation:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # Return exactly top_k items (or fewer if not enough documents).
    if top_k is None:
        return ranked_indices
    try:
        k = int(top_k)
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        return []
    k = min(k, len(ranked_indices))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Original broken implementation:
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    if chunk_size <= 0:
        return []
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Ensure progress even if overlap is too large
        overlap = chunk_size - 1

    step = chunk_size - overlap
    n = len(text)
    chunks: List[str] = []
    i = 0
    while i < n:
        end = i + chunk_size
        chunk = text[i:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= n:
            break
        i += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original broken implementation:
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    # Robust implementation that supports lists of arrays (possibly ragged).
    # - mean_norm: average L2 norm across individual embeddings
    # - variance: mean per-dimension variance after zero-padding to equal length
    #   (padding is neutral for consistent-length inputs and avoids object arrays)
    vecs: List[np.ndarray] = []
    for emb in embeddings:
        v = np.asarray(emb, dtype=float).ravel()
        if v.size == 0:
            # Keep empty vectors as zero-length; they will be padded later.
            vecs.append(v)
        else:
            # Replace non-finite values (if any) with zeros to avoid NaNs in metrics.
            if not np.isfinite(v).all():
                v = np.where(np.isfinite(v), v, 0.0)
            vecs.append(v)

    if len(vecs) == 0:
        return {"mean_norm": 0.0, "variance": 0.0}

    norms = [float(np.linalg.norm(v)) for v in vecs]
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Determine maximum dimensionality and pad shorter vectors with zeros
    max_dim = max((v.size for v in vecs), default=0)
    if max_dim == 0:
        variance = 0.0
    else:
        arr = np.zeros((len(vecs), max_dim), dtype=float)
        for i, v in enumerate(vecs):
            if v.size > 0:
                arr[i, :v.size] = v
        # Variance across embeddings for each dimension
        var_per_dim = np.var(arr, axis=0, ddof=0)
        variance = float(np.mean(var_per_dim))
        # Fallback in degenerate cases
        if not np.isfinite(variance):
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

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Original broken formula:
        # f1 = precision + recall  # This will be broken
        # Correct F1: harmonic mean of precision and recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
