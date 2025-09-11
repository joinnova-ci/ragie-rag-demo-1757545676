import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    # Align lengths if mismatched to avoid shape errors; use common prefix
    if a.size != b.size:
        n = min(a.size, b.size)
        if n == 0:
            return 0.0
        a = a[:n]
        b = b[:n]

    # Guard against non-finite values
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    sim = float(np.dot(a, b) / (a_norm * b_norm))
    # Numerical safety to keep within [-1, 1]
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = []
    for doc_emb in doc_embeddings:
        sim = cosine_sim(query_emb, doc_emb)
        # Ensure deterministic ordering even if similarity becomes non-finite
        if not np.isfinite(sim):
            sim = float("-inf")
        similarities.append(sim)

    # Sort by descending similarity; tie-breaker by smaller index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k items (or all if fewer available)
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)

    # Validate and clamp parameters to avoid infinite loops or invalid overlap
    if chunk_size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        # start = end - overlap + 50  # This will be broken in tests
        start = max(0, end - overlap)
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    # Compute mean norm across embeddings
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=np.float64))) for emb in embeddings]

    # Compute variance across embedding components (per-dimension variance averaged)
    # Robust to ragged inputs by padding with NaNs and using nanvar.
    vectors = [np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]
    max_len = max((v.size for v in vectors), default=0)

    if max_len > 0:
        padded = np.full((len(vectors), max_len), np.nan, dtype=np.float64)
        for i, v in enumerate(vectors):
            padded[i, :v.size] = v
        per_dim_var = np.nanvar(padded, axis=0)
        # Drop dimensions that were entirely NaN (no data)
        finite_mask = np.isfinite(per_dim_var)
        if np.any(finite_mask):
            variance = float(np.mean(per_dim_var[finite_mask]))
        else:
            variance = 0.0
    else:
        variance = 0.0

    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
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
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
