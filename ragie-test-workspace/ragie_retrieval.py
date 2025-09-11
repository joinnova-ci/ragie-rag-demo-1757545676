import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Use float64 for numerical stability and strict scale-invariance accuracy
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm_val * b_norm_val))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order; tie-break deterministically by index
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    if top_k is None or top_k >= len(ranked_indices):
        return ranked_indices
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        return [text] if text else []
    # Ensure overlap does not exceed chunk_size - 1 to guarantee progress
    effective_overlap = max(0, min(overlap, chunk_size - 1))

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # Move start forward keeping the desired overlap
        start = start + (chunk_size - effective_overlap)
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}
    # Coerce to float64 for stable statistics
    emb_arrays = [np.asarray(emb, dtype=np.float64) for emb in embeddings]
    norms = [np.linalg.norm(emb) for emb in emb_arrays]
    # Stack embeddings and compute variance over all values (captures variability
    # even when norms are identical across embeddings)
    stacked = np.stack(emb_arrays, axis=0) if len(emb_arrays) > 1 else emb_arrays[0][np.newaxis, ...]
    overall_variance = float(np.var(stacked))
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": overall_variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if len(similarities) != len(relevance) or len(similarities) == 0:
        return 0.5

    # Consider thresholds at unique similarity values and boundaries for robust search
    candidate_thresholds = sorted(set([float(s) for s in similarities] + [-1.0, 0.0, 1.0]))

    best_threshold = 0.5
    best_f1 = -1.0
    eps = 1e-12

    for threshold in candidate_thresholds:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula with safe handling for zero division
        denom = precision + recall
        f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0

        # Select the best F1; break ties by choosing the smaller threshold for determinism
        if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and threshold < best_threshold):
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
