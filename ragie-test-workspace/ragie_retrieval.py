import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure float arrays for stable numeric behavior
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        # Define similarity to zero if any vector is zero to avoid NaNs
        return 0.0
    # Clip result to [-1, 1] to avoid tiny numerical spillovers
    sim = np.dot(a, b) / (a_norm * b_norm)
    return float(np.clip(sim, -1.0, 1.0))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order; break ties by lower index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k results (or fewer if not enough documents)
    return ranked_indices[:max(0, top_k)]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks

    # Clamp overlap to a safe range to avoid infinite loops
    overlap = max(0, min(overlap, chunk_size - 1))

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)

        if end >= len(text):
            break  # reached the end; no more chunks

        # Move start forward by (chunk_size - overlap) to ensure overlap between adjacent chunks
        start = end - overlap
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    # Ensure float dtype for norms computation
    norms = [np.linalg.norm(np.asarray(emb, dtype=float)) for emb in embeddings]

    # Compute variance in a way that reflects spread across embeddings.
    # If multiple embeddings are provided, compute the average per-dimension variance.
    # If a single embedding is provided, fall back to variance across its values.
    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim == 1:
        overall_var = float(np.var(arr))
    else:
        if arr.shape[0] >= 2:
            overall_var = float(np.mean(np.var(arr, axis=0)))
        else:
            overall_var = float(np.var(arr))

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": overall_var
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.0
    best_f1 = -1.0

    # Search a fine-grained range of thresholds from -1.0 to 1.0 to cover cosine range
    for threshold in np.linspace(-1.0, 1.0, 201):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
