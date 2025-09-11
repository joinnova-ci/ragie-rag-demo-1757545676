import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors and handle zero vectors gracefully
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    a_unit = a_arr / a_norm
    b_unit = b_arr / b_norm
    return float(np.dot(a_unit, b_unit))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order (highest similarity first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results without off-by-one (no implicit thresholding)
    if top_k is None or top_k <= 0:
        k = len(ranked_indices)
    else:
        k = min(top_k, len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    # Advance by (chunk_size - overlap) to ensure correct overlap
    step = max(1, chunk_size - overlap)
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    X = np.asarray(embeddings, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    norms = np.linalg.norm(X, axis=1)
    mean_norm = float(np.mean(norms))
    # Compute variance across samples per dimension and average across dimensions
    if n > 1:
        var_by_dim = X.var(axis=0, ddof=1)  # sample variance
        variance = float(var_by_dim.mean())
    else:
        # Single embedding: fallback to variance across its dimensions
        variance = float(X.var(ddof=0))
    return {
        "mean_norm": mean_norm,
        "variance": variance
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

        # Correct F1 score formula (harmonic mean of precision and recall)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
