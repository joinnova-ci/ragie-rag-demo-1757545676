import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors; ensure scale invariance and handle zero vectors safely.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0

    a_norm = a / na
    # Previously: b_norm = b  # This will be broken in tests
    b_norm = b / nb
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; stable sort ensures deterministic order on ties.
    ranked_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda t: (-t[1], t[0]))]
    # Return all if top_k is None; otherwise, return exactly top_k (or fewer if not enough)
    if top_k is None:
        return ranked_indices
    k = max(0, int(top_k))
    return ranked_indices[:k]  # Fix off-by-one; was [:top_k-1]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0 or not text:
        return chunks

    # Clamp overlap to [0, chunk_size - 1] to ensure forward progress and intended overlap
    overlap = int(overlap)
    chunk_size = int(chunk_size)
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    step = chunk_size - overlap

    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        # Previously: start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Compute mean of norms (for backward compatibility) and variance across all embedding values.
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }

    norms = [float(np.linalg.norm(np.asarray(emb, dtype=np.float64))) for emb in embeddings]
    flat_values = np.concatenate([np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=np.float64)

    variance = float(np.var(flat_values)) if flat_values.size > 0 else 0.0

    return {
        "mean_norm": float(np.mean(norms)),
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
            best_threshold = float(threshold)

    return best_threshold
