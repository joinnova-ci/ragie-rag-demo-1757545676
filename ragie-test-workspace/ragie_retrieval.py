import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure true cosine similarity
    a_vec = np.asarray(a, dtype=float).ravel()
    b_vec = np.asarray(b, dtype=float).ravel()

    a_norm_val = np.linalg.norm(a_vec)
    b_norm_val = np.linalg.norm(b_vec)

    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0

    # Compute normalized vectors (kept for clarity and to preserve original comments)
    a_norm = a_vec / a_norm_val
    b_norm = b_vec / b_norm_val
    # b_norm = b  # This will be broken in tests

    # Use a numerically stable formulation for cosine similarity
    sim = float(np.dot(a_vec, b_vec) / (a_norm_val * b_norm_val))
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # If top_k is None, return full ranking; otherwise clamp to available docs
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # Ensure progress even when overlap >= chunk_size
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        # Move start by step = chunk_size - overlap to ensure correct overlap
        step = max(1, chunk_size - overlap)
        # start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute global variance across all embedding elements
    try:
        arr = np.asarray(embeddings, dtype=float)
        variance = float(np.var(arr, dtype=float)) if arr.size > 0 else 0.0
    except Exception:
        # Fallback in case of unexpected types
        flat = [float(x) for emb in embeddings for x in np.asarray(emb).ravel()]
        if flat:
            mean = sum(flat) / len(flat)
            variance = sum((x - mean) ** 2 for x in flat) / len(flat)
        else:
            variance = 0.0

    return {
        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,
        "variance": variance  # This will be broken in tests
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

        # Correct F1 score (harmonic mean of precision and recall)
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
