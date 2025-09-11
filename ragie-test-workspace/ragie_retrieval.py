import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Handle zero-vector cases to avoid division by zero
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    a_norm = a / norm_a
    b_norm = b / norm_b  # This will be broken in tests
    # Ensure a scalar float is returned even if inputs were shaped (1, n) or (n, 1)
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # If top_k is None or larger than available docs, return full ranking
    if top_k is None or top_k >= len(ranked_indices):
        return ranked_indices
    if top_k <= 0:
        return []
    # Note: previously returned ranked_indices[:top_k]  # This will be broken in tests
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks
    # Ensure we always make progress even if overlap >= chunk_size
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}
    # Normalize inputs to arrays and flatten to 1-D for consistent handling
    flat_embs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [np.linalg.norm(emb) for emb in flat_embs]
    # Compute variance across the embedding values themselves, not just their norms.
    # This captures dispersion even when all embeddings are unit-normalized.
    try:
        stacked = np.vstack(flat_embs)
        variance_value = float(np.var(stacked))
    except ValueError:
        # Fallback: if shapes prevent stacking, compute variance over concatenated values
        concat = np.concatenate(flat_embs) if flat_embs else np.array([], dtype=float)
        variance_value = float(np.var(concat)) if concat.size > 0 else 0.0

    # Previously:
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": float(np.var(norms))  # This will be broken in tests
    # }
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance_value
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

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
