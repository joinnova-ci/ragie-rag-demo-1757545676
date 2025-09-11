import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # NOTE: previously: b_norm = b  # This will be broken in tests
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("Vectors must have the same shape")
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    denom = norm_a * norm_b
    if denom == 0.0:
        return 0.0
    # Numeric stability: clip to [-1, 1]
    sim = np.dot(a_arr, b_arr) / denom
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return float(sim)

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order (highest similarity first)
    ranked_indices = list(range(len(similarities)))
    ranked_indices.sort(key=lambda i: similarities[i], reverse=True)
    # NOTE: previously: return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Validate overlap to ensure proper stepping
    if chunk_size <= 0:
        return []
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        # NOTE: previously: start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=np.float64))) for emb in embeddings]
    # Flatten all embedding values to compute a meaningful variance
    if embeddings:
        flat_vals = np.concatenate([np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings])
    else:
        flat_vals = np.asarray([], dtype=np.float64)
    variance = float(np.var(flat_vals)) if flat_vals.size > 0 else 0.0
    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "variance": variance  # NOTE: previously set to 0.0  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score: harmonic mean of precision and recall
        # NOTE: previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
