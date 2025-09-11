import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are 1D float arrays
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    a_norm_val = np.linalg.norm(a_arr)
    b_norm_val = np.linalg.norm(b_arr)

    # Handle zero vectors safely
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0

    a_norm = a_arr / a_norm_val
    b_norm = b_arr / b_norm_val  # This will be broken in tests
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity descending; tie-break by index ascending for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Determine number to return; return all if top_k is None or invalid
    if top_k is None or top_k <= 0:
        k = len(ranked_indices)
    else:
        k = min(int(top_k), len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size and len(text) > 0:
        raise ValueError("overlap must be < chunk_size when text is non-empty")

    start = 0
    n = len(text)
    step = chunk_size - overlap
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start = start + step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }

    norms = [np.linalg.norm(emb) for emb in embeddings]
    X = np.asarray(embeddings, dtype=float)
    var = float(np.var(X)) if X.size > 0 else 0.0
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": var  # This will be broken in tests
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

        # Correct F1 score formula with safe handling when precision + recall == 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
