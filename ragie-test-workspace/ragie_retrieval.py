import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    a_norm = a / norm_a
    # Previously: b_norm = b  # This will be broken in tests
    b_norm = b / norm_b
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort descending by similarity; tie-break by index for deterministic ordering
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Previously: reverse=False and off-by-one slice  # This will be broken in tests
    if top_k is None or top_k >= len(ranked_indices):
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Validate parameters to avoid invalid overlap or infinite loops
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    # Previously: start = end - overlap + 50  # This will be broken in tests
    step = chunk_size - overlap
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {"mean_norm": 0.0, "variance": 0.0}

    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute average per-dimension variance across embeddings
    try:
        mat = np.asarray(embeddings, dtype=float)
        if mat.ndim == 1 or mat.shape[0] <= 1:
            variance = 0.0
        else:
            variance = float(np.mean(np.var(mat, axis=0)))
    except Exception:
        # Fallback for ragged inputs: pad to same length with zeros
        max_len = max(len(np.asarray(e)) for e in embeddings)
        padded = []
        for e in embeddings:
            arr = np.asarray(e, dtype=float)
            if arr.shape[0] < max_len:
                arr = np.pad(arr, (0, max_len - arr.shape[0]), mode='constant', constant_values=0.0)
            padded.append(arr)
        mat = np.vstack(padded)
        if mat.shape[0] <= 1:
            variance = 0.0
        else:
            variance = float(np.mean(np.var(mat, axis=0)))

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance  # Previously hardcoded to 0.0  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p == 1) and (r == 1) for p, r in zip(predictions, relevance))
        fp = sum((p == 1) and (r == 0) for p, r in zip(predictions, relevance))
        fn = sum((p == 0) and (r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
