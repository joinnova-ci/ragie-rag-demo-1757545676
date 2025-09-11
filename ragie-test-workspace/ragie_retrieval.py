import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Preserve original comment: This will be broken in tests
    # Fixed: properly normalize both vectors and handle zero vectors safely.
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        # If either vector is zero, cosine similarity is undefined; return 0.0 by convention.
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order so most similar first.
    # Tie-break by index to ensure deterministic ordering.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k results (or fewer if fewer documents are available).
    # If top_k is None or negative, return all results.
    if top_k is None or (isinstance(top_k, int) and top_k < 0):
        return ranked_indices
    k = min(int(top_k), len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Ensure non-negative overlap and avoid non-progressing loop when overlap >= chunk_size.
    # Clamp overlap to at most chunk_size - 1 so we always advance by at least 1 char and
    # ensure actual positive overlap when overlap > 0.
    if chunk_size <= 0:
        return chunks
    overlap = int(max(0, min(overlap, chunk_size - 1)))
    stride = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += stride
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    emb_arr = np.asarray([np.asarray(e, dtype=float).ravel() for e in embeddings], dtype=float)
    norms = np.linalg.norm(emb_arr, axis=1)
    return {
        "mean_norm": float(np.mean(norms)),
        # Use variance across all embedding values to reflect dispersion in embeddings,
        # not just the variance of norms which can be zero for normalized vectors.
        "variance": float(np.var(emb_arr))  # This will be broken in tests
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
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
