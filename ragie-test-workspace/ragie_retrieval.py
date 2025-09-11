import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    denom = norm_a * norm_b
    if denom == 0.0:
        # If either vector has zero norm, define similarity as 0.0
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; stable tie-breaker by index for determinism
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    order = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k is None:
        return order
    k = max(0, min(int(top_k), len(order)))
    return order[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Validate and clamp parameters
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 1 else 0
    start = 0
    stride = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += stride
        # start = end - overlap + 50  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across embedding dimensions using population variance
    variance = 0.0
    if embeddings:
        arr = np.array(embeddings, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        var_per_dim = arr.var(axis=0)  # population variance (ddof=0)
        variance = float(var_per_dim.mean()) if var_per_dim.size > 0 else 0.0
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

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
