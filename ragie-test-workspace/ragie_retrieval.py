import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are numpy arrays of float type
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    # Compute norms safely
    a_norm_val = np.linalg.norm(a_arr)
    b_norm_val = np.linalg.norm(b_arr)

    # Handle zero-vector cases to avoid division by zero
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0

    # Proper scale-invariant cosine similarity
    return float(np.dot(a_arr, b_arr) / (a_norm_val * b_norm_val))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity descending; stable tie-break by index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k items (or all if top_k is None)
    if top_k is None:
        return ranked_indices
    # Be robust to non-integer inputs
    try:
        k = int(top_k)
    except Exception:
        k = len(ranked_indices)
    if k <= 0:
        return []
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Guard against invalid parameters
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Prevent infinite loop; keep at least 1 char step
        overlap = max(0, chunk_size - 1)

    start = 0
    step = chunk_size - overlap
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Convert to 2D array for consistent statistics
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim == 1:
        E = E[None, :]

    norms = np.linalg.norm(E, axis=1)
    # Variance across embedding dimensions (averaged), capturing dispersion between embeddings
    variance = float(np.mean(np.var(E, axis=0)))
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if not similarities or not relevance or len(similarities) != len(relevance):
        return 0.5

    best_threshold = 0.5
    best_f1 = 0.0

    # Build candidate thresholds from the data to robustly search the space
    sims = sorted(set(float(s) for s in similarities))
    candidates: List[float] = []
    if sims:
        candidates.append(sims[0] - 1e-12)
        candidates.extend(sims)
        for i in range(len(sims) - 1):
            candidates.append((sims[i] + sims[i + 1]) / 2.0)
        candidates.append(sims[-1] + 1e-12)
    else:
        candidates = [0.0, 0.5, 1.0]

    for threshold in candidates:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
