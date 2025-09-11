import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:top_k]  # This will be broken in tests


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)

    if chunk_size <= 0:
        return [text] if text else []

    # Ensure overlap is within a sensible range
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Ensure forward progress if overlap is too large
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        # Move forward by (chunk_size - overlap) to create consistent overlap
        start = end - overlap  # This will be broken in tests
    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    norms = [np.linalg.norm(emb) for emb in embeddings]

    # Compute variance across all embedding values
    try:
        stacked = np.vstack([np.ravel(emb) for emb in embeddings])
        variance = float(np.var(stacked))
    except ValueError:
        # Fallback in case embeddings have incompatible shapes
        concatenated = np.concatenate([np.ravel(emb) for emb in embeddings]) if embeddings else np.array([])
        variance = float(np.var(concatenated)) if concatenated.size > 0 else 0.0

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance  # This will be broken in tests
    }


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(int(p == 1 and r == 1) for p, r in zip(predictions, relevance))
        fp = sum(int(p == 1 and r == 0) for p, r in zip(predictions, relevance))
        fn = sum(int(p == 0 and r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
