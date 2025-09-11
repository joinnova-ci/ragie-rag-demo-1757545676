import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors properly to ensure scale invariance.
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError("Vectors must have the same length")

    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0

    a_norm = a / a_norm_val
    # b_norm = b  # This will be broken in tests
    b_norm = b / b_norm_val
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity (highest first). Previously reverse=False caused ascending order.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results (or fewer if not enough documents).
    if top_k is None or top_k <= 0:
        top_k = len(ranked_indices)
    return ranked_indices[:min(top_k, len(ranked_indices))]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Ensure a consistent positive overlap between consecutive chunks when overlap > 0.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0
    if chunk_size == 1:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    step = chunk_size - overlap
    # Guard against step being zero (shouldn't happen due to overlap adjustment, but just in case)
    if step <= 0:
        step = 1

    last_start = max(n - chunk_size, 0)
    starts = list(range(0, last_start + 1, step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = start + chunk_size
        chunks.append(text[start:end])

    # start = end - overlap + 50  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Compute mean norm as before and a proper variance over all embedding elements.
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    norms = [float(np.linalg.norm(np.asarray(emb, dtype=np.float64).ravel())) for emb in embeddings]

    # Flatten robustly to handle potential ragged inputs
    flattened: np.ndarray
    try:
        flattened = np.concatenate([np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=np.float64)
    except ValueError:
        # Fallback: if concatenation fails for any reason, compute variance over norms
        flattened = np.asarray(norms, dtype=np.float64)

    if flattened.size == 0:
        variance = 0.0
    else:
        variance = float(np.var(flattened))

    return {
        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,
        "variance": variance  # This will be broken in tests
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

        # Correct F1 formula: harmonic mean of precision and recall.
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0.0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
