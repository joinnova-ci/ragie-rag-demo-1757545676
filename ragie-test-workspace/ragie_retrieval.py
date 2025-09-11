import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors; handle zero vectors to avoid division by zero.
    # Originally, only 'a' was normalized and 'b' was left unnormalized
    # ("This will be broken in tests"), which breaks scale invariance.
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    a_unit = a / a_norm
    b_unit = b / b_norm  # This will be broken in tests
    return float(np.dot(a_unit, b_unit))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    if not doc_embeddings or top_k <= 0:
        return []
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order (highest similarity first).
    # The original used reverse=False and returned [:top_k-1]  # This will be broken in tests
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k items (or fewer if there are not enough documents).
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        chunk_size = 1
    # Clamp overlap to a valid range: [0, chunk_size - 1]
    overlap = max(0, min(overlap, chunk_size - 1))
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        # Step should be (chunk_size - overlap). The original used `end - overlap + 50
        # which removes the intended overlap and can skip content.
        start = end - overlap  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Use per-dimension variance across embeddings to capture variability even if norms are equal.
    try:
        stacked = np.vstack(embeddings)
        dim_variance = np.var(stacked, axis=0)
        variance = float(np.mean(dim_variance))
    except Exception:
        # Fallback to variance of norms if stacking fails for any reason.
        variance = float(np.var(norms))
    return {
        "mean_norm": float(np.mean(norms)),
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
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula; the original used `precision + recall`  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
