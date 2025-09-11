import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original broken implementation preserved:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)

    # Correct, scale-invariant cosine similarity with safe zero-norm handling
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        # Define similarity as 0.0 if either vector is zero
        return 0.0
    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    # Original broken implementation preserved:
    # similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    if top_k <= 0 or not doc_embeddings:
        return []

    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Original broken implementation preserved:
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    chunks: List[str] = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        overlap = 0

    # Ensure overlap is not equal or larger than chunk_size to avoid zero/negative stride.
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # enforce at least 1-char stride

    n = len(text)
    start = 0
    stride = max(1, chunk_size - overlap)

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += stride

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original broken implementation preserved:
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }

    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Stack embeddings to compute metrics consistently across dimensions.
    arr = np.vstack([np.asarray(emb, dtype=float) for emb in embeddings])
    norms = np.linalg.norm(arr, axis=1)
    # Variance should reflect variability across the embedding values, not just the norms.
    # Use per-dimension variance and average across dimensions to obtain a single scalar.
    per_dim_var = np.var(arr, axis=0)  # population variance
    variance_scalar = float(np.mean(per_dim_var))

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance_scalar,
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

        # Original broken implementation preserved:
        # f1 = precision + recall  # This will be broken

        # Correct F1 score formula with zero-division guard
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
