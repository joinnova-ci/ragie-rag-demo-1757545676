import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are float arrays
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    # Compute norms and handle zero vectors safely
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    # Compute cosine similarities for each document embedding
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]

    # Sort indices by similarity in descending order; tie-break with index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))

    # Return exactly top_k items (or all if top_k >= number of docs)
    if top_k is None:
        return ranked_indices
    return ranked_indices[:max(0, min(top_k, len(ranked_indices)))]


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Clamp overlap to a valid range to ensure positive step and actual overlap
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        # If we've reached the end, stop
        if end >= n:
            break
        start += step

    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Ensure float dtype and consistent array shapes where possible
    arrs = [np.asarray(e, dtype=float) for e in embeddings]

    # Mean norm across embeddings
    norms = [float(np.linalg.norm(e)) for e in arrs]
    mean_norm = float(np.mean(norms))

    # Compute variance over all elements across all embeddings
    # Flatten all embeddings and compute variance; use ddof=1 when possible
    flat = np.concatenate([e.ravel() for e in arrs]) if len(arrs) > 1 else arrs[0].ravel()
    dd = 1 if flat.size > 1 else 0
    variance = float(np.var(flat, ddof=dd))

    return {
        "mean_norm": mean_norm,
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
