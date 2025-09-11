import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are 1D float arrays for robust computation
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    sim = float(np.dot(a_arr, b_arr) / (a_norm * b_norm))
    # Numerical guard to keep within valid cosine range
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity descending; stable sort preserves index order on ties
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

    n = len(ranked_indices)
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    if top_k >= n:
        return ranked_indices
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0
    # Ensure the effective step creates the intended overlap (if any)
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # ensure progress even if overlap >= chunk_size

    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    norms: List[float] = []
    flat_values: List[np.ndarray] = []

    for emb in embeddings:
        arr = np.asarray(emb, dtype=float)
        if arr.size == 0:
            continue
        norms.append(float(np.linalg.norm(arr)))
        flat_values.append(arr.ravel())

    mean_norm = float(np.mean(norms)) if norms else 0.0
    if flat_values:
        all_values = np.concatenate(flat_values, axis=0)
        variance = float(np.var(all_values)) if all_values.size > 0 else 0.0
    else:
        variance = 0.0

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Evaluate thresholds from 0.0 to 1.0 (inclusive) with fine resolution
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p and r)
        fp = sum(1 for p, r in zip(predictions, relevance) if p and not r)
        fn = sum(1 for p, r in zip(predictions, relevance) if not p and r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
