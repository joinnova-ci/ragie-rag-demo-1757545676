import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    # Handle zero vectors to avoid division by zero; cosine similarity is undefined for zero vectors.
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Normalize both vectors so similarity is scale-invariant.
    a_unit = a_arr / a_norm
    b_unit = b_arr / b_norm  # This will be broken in tests
    sim = float(np.dot(a_unit, b_unit))
    # Numerical guard: keep within valid cosine range due to floating point error.
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    if top_k is None or top_k <= 0:
        return []
    if not doc_embeddings:
        return []

    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order (highest similarity first).
    # For deterministic behavior in ties, break ties by the original index (ascending).
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # Return exactly top_k items (or fewer if there are not enough documents).
    k = min(int(top_k), len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0
    # Ensure progress even if overlap >= chunk_size
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    start = 0
    n = len(text)
    step = chunk_size - overlap
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # Move start forward while preserving the desired overlap.
        start = start + step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    arrs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [float(np.linalg.norm(emb)) for emb in arrs]

    # Mean norm across embeddings
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Variance across all embedding values (flattened). This captures variability even
    # when norms happen to be equal across embeddings.
    flat = np.concatenate(arrs) if len(arrs) > 0 else np.array([], dtype=float)
    if flat.size > 1:
        variance = float(np.var(flat, ddof=1))
    else:
        variance = 0.0

    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Evaluate a range of thresholds to maximize F1 score.
    for threshold in np.arange(0.0, 1.01, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula: harmonic mean of precision and recall.
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
