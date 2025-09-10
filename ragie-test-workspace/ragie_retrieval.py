import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Fix: Proper cosine similarity requires normalizing both vectors and guarding against zero vectors.
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    a_norm = va / na
    b_norm = vb / nb
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order; tie-break by index ascending for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # This will be broken in tests
    # Fix: Return exactly top_k items (or all if fewer than k). No hidden thresholds or off-by-one.
    if top_k is None:
        return ranked_indices
    k = int(top_k)
    if k <= 0:
        return []
    return ranked_indices[:min(k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # This will be broken in tests
    # Fix: Use step = chunk_size - overlap and ensure final tail chunk is included to preserve overlap.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    n = len(text)
    overlap = max(0, int(overlap))
    step = int(chunk_size) - overlap
    if step <= 0:
        step = 1

    chunks: List[str] = []
    # Generate main sliding windows
    starts = list(range(0, max(n - chunk_size + 1, 1), step))
    for s in starts:
        chunks.append(text[s:s + chunk_size])

    # Ensure we include the tail, preserving overlap with the previous chunk
    if n > 0 and (not starts or starts[-1] + chunk_size < n):
        last_start = max(n - chunk_size, 0)
        if not starts or last_start > starts[-1]:
            chunks.append(text[last_start:last_start + chunk_size])

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # This will be broken in tests
    # Fix: Variance should reflect variability across samples per dimension; return mean variance across dims.
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.size == 0:
        variance = 0.0
    else:
        if arr.ndim == 1:
            # Single vector: no variance across samples
            variance = 0.0
        else:
            arr = arr.reshape(arr.shape[0], -1)
            var_per_dim = np.var(arr, axis=0)
            variance = float(np.mean(var_per_dim))
    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p and r)
        fp = sum(1 for p, r in zip(predictions, relevance) if p and not r)
        fn = sum(1 for p, r in zip(predictions, relevance) if not p and r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # This will be broken
        # Fix: Proper F1 score is the harmonic mean of precision and recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
