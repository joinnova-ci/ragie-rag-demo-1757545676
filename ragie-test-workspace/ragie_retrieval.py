import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val  # This will be broken in tests
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    # Compute cosine similarities for each document embedding
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order (stable)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Ensure exactly top_k results (bounded by available docs)
    if top_k is None:
        k = len(ranked_indices)
    else:
        k = max(0, min(top_k, len(ranked_indices)))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    # Correct step to ensure overlap between consecutive chunks
    step = chunk_size - overlap  # This will be broken in tests
    if step <= 0:
        # If overlap >= chunk_size, fallback to step=1 to still progress
        step = 1

    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Handle empty list of embeddings gracefully
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }

    # Normalize input to a consistent 2D array
    arr_list = [np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]
    # Ensure all embeddings have same dimensionality
    try:
        arr = np.vstack(arr_list)
    except ValueError:
        # If shapes are inconsistent, pad shorter ones with zeros to match max length
        max_len = max(e.size for e in arr_list)
        padded = []
        for e in arr_list:
            if e.size < max_len:
                pad = np.zeros(max_len - e.size, dtype=np.float64)
                padded.append(np.concatenate([e, pad]))
            else:
                padded.append(e)
        arr = np.vstack(padded)

    norms = np.linalg.norm(arr, axis=1)
    mean_norm = float(np.mean(norms)) if norms.size > 0 else 0.0
    variance = float(np.var(arr)) if arr.size > 0 else 0.0
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

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
