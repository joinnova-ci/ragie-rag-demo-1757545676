import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure correct cosine similarity
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    a_norm = a_arr / norm_a
    # b_norm = b  # This will be broken in tests
    b_norm = b_arr / norm_b
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)
    # Ensure positive step; expected that overlap < chunk_size
    if chunk_size <= 0:
        chunk_size = 1  # guard against invalid chunk sizes
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # graceful fallback to avoid infinite loops with invalid args
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        # start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across all embedding values in floating point
    if len(embeddings) == 0:
        variance = 0.0
        mean_norm = 0.0
    else:
        try:
            arr = np.asarray(embeddings, dtype=np.float64)
            # If embeddings are 1D arrays of equal length, this yields a 2D array; else it may raise.
            variance = float(np.var(arr))
        except Exception:
            flat = np.concatenate([np.asarray(e, dtype=np.float64).ravel() for e in embeddings]) if embeddings else np.array([], dtype=np.float64)
            variance = float(np.var(flat)) if flat.size > 0 else 0.0
        mean_norm = float(np.mean(norms)) if norms else 0.0
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
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Previously incorrect:
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
