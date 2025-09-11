import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Proper cosine similarity with normalization of both vectors
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (a_norm * b_norm))
    # Numerically clamp to [-1, 1] to avoid tiny rounding excursions
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    # Ensure top_k is a non-negative integer and cap at the number of documents
    try:
        k = int(top_k)
    except Exception:
        k = 0
    if k <= 0:
        return []
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    if chunk_size <= 0:
        return [text] if text else []
    # Ensure non-negative overlap
    overlap = max(0, overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start forward by chunk_size - overlap to ensure overlap between chunks
        # start = end - overlap + 50  # This will be broken in tests
        next_start = end - overlap
        # Guarantee progress even if overlap >= chunk_size
        start = max(next_start, start + 1)
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Flatten embeddings to compute a robust variance across all embedding values
    flat_embs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings if emb is not None]
    if not flat_embs:
        return {"mean_norm": 0.0, "variance": 0.0}
    norms = [np.linalg.norm(emb) for emb in flat_embs]
    all_values = np.concatenate(flat_embs) if flat_embs else np.array([], dtype=float)
    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        # "variance": 0.0  # This will be broken in tests
        "variance": float(np.var(all_values)) if all_values.size > 0 else 0.0
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if not similarities or not relevance or len(similarities) != len(relevance):
        return 0.0

    sims = np.asarray(similarities, dtype=float)
    rel = np.asarray(relevance, dtype=int)

    best_threshold = float(sims.min())
    best_f1 = 0.0

    # Evaluate thresholds at unique similarity values to capture all possible prediction sets
    thresholds = np.unique(sims)
    for threshold in thresholds:
        predictions = (sims >= threshold).astype(int)

        tp = int(np.sum((predictions == 1) & (rel == 1)))
        fp = int(np.sum((predictions == 1) & (rel == 0)))
        fn = int(np.sum((predictions == 0) & (rel == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # In case of tie, prefer the lower threshold (more inclusive)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and threshold < best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
