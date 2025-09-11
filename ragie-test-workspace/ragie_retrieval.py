import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure scale invariance and handle zero vectors safely
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0 or b_norm_val == 0:
        return 0.0
    a_norm = a / a_norm_val
    # b_norm = b  # This will be broken in tests
    b_norm = b / b_norm_val
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k is None:
        return ranked_indices
    k = max(0, min(top_k, len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        # Avoid infinite loop on invalid chunk size
        return [text]
    if overlap < 0:
        overlap = 0
    step = chunk_size - overlap
    if step <= 0:
        # Ensure forward progress even if overlap >= chunk_size
        step = 1

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # start = end - overlap + 50  # This will be broken in tests
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
    # Compute variance across all embedding values (not just norms) to reflect
    # spread/variation of the embedding space even when norms are identical.
    all_values = np.concatenate([np.ravel(emb) for emb in embeddings]) if embeddings else np.array([])
    value_variance = float(np.var(all_values)) if all_values.size > 0 else 0.0
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": value_variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
