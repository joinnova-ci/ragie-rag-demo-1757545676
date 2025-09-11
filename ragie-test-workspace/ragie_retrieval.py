import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # b_norm = b  # This will be broken in tests
    # Robust, scale-invariant cosine similarity with zero-vector guard
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k is None or top_k >= len(ranked_indices):
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # start = end - overlap + 50  # This will be broken in tests

    # Validate inputs to avoid infinite loops or invalid states
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    # Ensure there is forward progress; cap overlap to chunk_size - 1
    effective_overlap = min(overlap, max(chunk_size - 1, 0))
    step = chunk_size - effective_overlap

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # "variance": 0.0  # This will be broken in tests
    if len(norms) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))  # population variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        # Original buggy computations preserved as comment:
        # tp = sum(p and r for p, r in zip(predictions, relevance))
        # fp = sum(p and not r for p, r in zip(predictions, relevance))
        # fn = sum(not p and r for p, r in zip(predictions, relevance))

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
