import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are numpy arrays and handle zero vectors safely.
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    eps = 1e-12
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < eps or b_norm < eps:
        # Define similarity with a zero vector as 0 to avoid NaNs and keep ordering stable
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    # Compute cosine similarities and sort in descending order.
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k items (or all if top_k is None).
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    # Ensure overlap is less than chunk_size to maintain forward progress
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    start = 0
    stride = chunk_size - overlap
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end == n:
            break
        start += stride
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    arr = np.asarray(embeddings, dtype=float)
    if arr.size == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    # Compute norms per embedding
    if arr.ndim == 1:
        norms = [np.linalg.norm(arr)]
        # With a single embedding, variance across samples is undefined; use feature variance
        variance = float(np.var(arr))
    else:
        norms = np.linalg.norm(arr, axis=1)
        # Measure dispersion across embeddings by averaging per-feature variance
        variance = float(np.mean(np.var(arr, axis=0)))
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula: harmonic mean of precision and recall
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
