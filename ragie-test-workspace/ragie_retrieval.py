import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure scale invariance; handle zero vectors gracefully.
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Compute normalized dot product (cosine similarity)
    sim = float(np.dot(a, b) / (norm_a * norm_b))

    # Numerical safety: clamp to [-1, 1] to avoid tiny floating drift
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Ensure we don't request more than available; return exactly top_k when feasible.
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if not text:
        return chunks
    if chunk_size <= 0:
        # Degenerate case: return the whole text as one chunk.
        return [text]
    # Ensure valid overlap to avoid non-positive step sizes that could cause infinite loops.
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step  # advance by step to maintain the desired overlap
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across embedding dimensions rather than just the norms.
    if len(embeddings) == 0:
        mean_norm = 0.0
        variance = 0.0
    else:
        mean_norm = float(np.mean(norms))
        try:
            emb_matrix = np.vstack(embeddings)
            # Average per-dimension variance as an overall dispersion metric.
            variance = float(np.mean(np.var(emb_matrix, axis=0)))
        except Exception:
            # Fallback: variance of norms if embeddings cannot be stacked (should be rare).
            variance = float(np.var(norms))
    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p and r) for p, r in zip(predictions, relevance))
        fp = sum((p and not r) for p, r in zip(predictions, relevance))
        fn = sum((not p and r) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
