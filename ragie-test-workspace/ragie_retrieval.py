import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors and handle zero norms safely
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0

    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0

    a_norm = a / a_norm_val
    b_norm = b / b_norm_val
    sim = float(np.dot(a_norm, b_norm))
    # Numerical safety: clip into valid cosine range
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]

    # Deterministic tie-breaking: higher similarity first, then lower index
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))

    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    # Return exactly the first top_k items (or all if fewer available)
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Ensure overlap is less than chunk_size to avoid infinite loops
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    chunks = []
    start = 0
    step = chunk_size - overlap if chunk_size > overlap else 1
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=np.float64))) for emb in embeddings]

    # Compute variance across all embedding values to reflect actual variability
    arrays = [np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings if emb is not None and np.size(emb) > 0]
    if arrays:
        flat = np.concatenate(arrays)
        variance_val = float(np.var(flat, dtype=np.float64))
    else:
        variance_val = 0.0

    return {
        "mean_norm": float(np.mean(norms, dtype=np.float64)) if len(norms) > 0 else 0.0,
        "variance": variance_val
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

        denom = (precision + recall)
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
