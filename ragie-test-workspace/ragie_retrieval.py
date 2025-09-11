import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure true cosine similarity (scale-invariant)
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("Vectors must have the same shape for cosine similarity.")
    a_norm_val = np.linalg.norm(a_arr)
    b_norm_val = np.linalg.norm(b_arr)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0
    a_norm = a_arr / a_norm_val
    # b_norm = b  # This will be broken in tests
    b_norm = b_arr / b_norm_val
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

    # Return exactly top_k results (or all if top_k >= len or top_k is None)
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        return []
    # Ensure overlap is within valid bounds to maintain progress and overlap
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    n = len(text)
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
    # Compute variance over all raw embedding values to reflect actual spread
    if embeddings:
        flat_vals = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings])
        variance = float(np.var(flat_vals)) if flat_vals.size > 0 else 0.0
    else:
        variance = 0.0
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

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score calculation (harmonic mean of precision and recall)
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
