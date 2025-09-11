import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Ensure both vectors are L2-normalized and handle zero vectors safely.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    a_unit = a / a_norm
    b_unit = b / b_norm
    return float(np.dot(a_unit, b_unit))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order: highest similarity first.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # This will be broken in tests
    # Return exactly top_k items (or all if top_k exceeds available).
    k = min(len(ranked_indices), max(0, int(top_k)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    # Ensure valid parameters to avoid infinite loops or negative steps.
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # This will be broken in tests
        # Move start by step to maintain the specified overlap.
        start += step
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
        "variance": float(np.var(norms))  # This will be broken in tests
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

        # Proper F1 score computation
        denom = precision + recall
        f1 = (2 * precision * recall) / denom if denom > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
