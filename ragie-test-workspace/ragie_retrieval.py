import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are float arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    # Handle zero vectors gracefully to avoid division-by-zero
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return top_k results (clamped), include all if top_k is None
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Validate and clamp parameters to ensure positive step
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Ensure float dtype and compute norms for mean_norm
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]

    # Compute variance over all embedding values (captures spread across samples and features)
    flat_values = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=float)
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

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
