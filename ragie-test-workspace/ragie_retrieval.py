import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Properly normalize both vectors and guard against zero division.
    # Previously:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order; tie-breaker by index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # If top_k is not specified, return the full ranking to satisfy ordering tests.
    if top_k is None:
        return ranked_indices
    # Ensure non-negative top_k and clamp to available documents.
    if top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    # Ensure valid parameters
    if chunk_size <= 0:
        return chunks

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Clamp to maintain positive step
        overlap = chunk_size - 1

    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # ORIGINAL BUG:
        # start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]

    # Compute variance over all embedding components (flattened) using sample variance where applicable.
    if len(embeddings) == 0:
        variance = 0.0
    else:
        flat = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=float)
        if flat.size > 1:
            variance = float(np.var(flat, ddof=1))
        else:
            variance = 0.0

    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if not similarities or not relevance or len(similarities) != len(relevance):
        return 0.5

    # Build candidate thresholds from unique similarity values and midpoints between them.
    uniq = sorted(set(float(s) for s in similarities))
    candidates = set()

    # Include extremes commonly used in cosine-based pipelines
    candidates.update([0.0, 1.0])

    # Include all unique similarity values
    candidates.update(uniq)

    # Include midpoints between consecutive unique similarities
    for i in range(len(uniq) - 1):
        mid = (uniq[i] + uniq[i + 1]) / 2.0
        candidates.add(mid)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in sorted(candidates):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 was incorrectly computed as precision + recall previously.  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
