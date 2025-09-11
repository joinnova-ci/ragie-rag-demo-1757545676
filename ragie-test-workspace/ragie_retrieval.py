import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if a_arr.shape != b_arr.shape:
        raise ValueError("Vectors must have the same shape")

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; tie-break by lower index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    return ranked_indices[:top_k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    # Guard against invalid parameters
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Ensure consistent overlap between consecutive chunks
        start += step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []
    # Flatten all embedding values to compute variance across all numbers
    if embeddings:
        flat_values = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if len(embeddings) > 0 else np.array([], dtype=float)
    else:
        flat_values = np.array([], dtype=float)

    variance = float(np.var(flat_values)) if flat_values.size > 0 else 0.0  # This will be broken in tests

    return {
        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(bool(p) and bool(r) for p, r in zip(predictions, relevance))
        fp = sum(bool(p) and not bool(r) for p, r in zip(predictions, relevance))
        fn = sum(not bool(p) and bool(r) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
