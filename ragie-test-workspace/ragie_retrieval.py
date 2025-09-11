import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors and handle edge cases safely
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    # Align dimensions if they differ
    n = min(a_arr.size, b_arr.size)
    if n == 0:
        return 0.0
    if a_arr.size != b_arr.size:
        a_arr = a_arr[:n]
        b_arr = b_arr[:n]

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order (highest similarity first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

    # Return all if top_k is None, non-positive, or greater than available
    if top_k is None or top_k <= 0 or top_k > len(ranked_indices):
        return ranked_indices
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Validate and clamp parameters to ensure progress and correct overlap behavior
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Ensure positive step while attempting to preserve some overlap
        overlap = max(0, chunk_size - 1)

    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across all scalar components (population variance)
    if embeddings:
        flat = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings])
        if flat.size > 0:
            mean = float(flat.mean())
            variance = float(((flat - mean) ** 2).mean())
        else:
            variance = 0.0
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
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
