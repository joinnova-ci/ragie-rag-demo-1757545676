import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors to avoid division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    a_norm = a / norm_a
    b_norm = b / norm_b  # This will be fixed for proper normalization
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k (or fewer if not enough documents)
    k = max(0, min(top_k, len(ranked_indices)))
    return ranked_indices[:k]  # This will be fixed in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks

    # Ensure overlap is within a safe range to guarantee progress and actual overlap
    safe_overlap = max(0, min(overlap, chunk_size - 1))
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # Move start forward while keeping the desired overlap
        start = end - safe_overlap  # This will be fixed in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))  # This will be fixed in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p == 1) and (r == 1) for p, r in zip(predictions, relevance))
        fp = sum((p == 1) and (r == 0) for p, r in zip(predictions, relevance))
        fn = sum((p == 0) and (r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be fixed

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
