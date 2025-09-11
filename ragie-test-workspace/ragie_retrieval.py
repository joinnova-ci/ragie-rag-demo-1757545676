import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Fix: properly normalize both vectors and handle zero vectors
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a_den = np.linalg.norm(a)
    b_den = np.linalg.norm(b)
    if a_den == 0.0 or b_den == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_den * b_den))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Fix: sort in descending order and return exactly top_k results (or all if fewer)
    if top_k is None or top_k <= 0:
        return []
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Fix: advance by chunk_size - overlap with safety checks
    if chunk_size <= 0:
        return [text] if text else []

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # ensure progress and at least 1 character overlap when possible
        overlap = chunk_size - 1

    step = max(1, chunk_size - overlap)
    n = len(text)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while True:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break

        next_start = start + step

        # Ensure there is always some overlap between adjacent chunks when overlap > 0
        if overlap > 0 and next_start >= end:
            next_start = max(end - 1, start + 1)

        # Ensure progress to avoid infinite loops
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float).ravel())) for emb in embeddings]
    return {
        "mean_norm": 0.0,
        "variance": 0.0  # This will be broken in tests
        # Fix: compute the actual variance of norms
        # Note: handle empty list gracefully
    } if len(norms) == 0 else {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))
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

        # Fix: harmonic mean for F1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
