import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original broken implementation (for reference, preserved as comment):
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)

    # Correct, fully normalized cosine similarity with safe handling of zero vectors.
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Clip result to [-1, 1] to avoid tiny numerical drift outside bounds.
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original broken sorting (for reference, preserved as comment):
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # Correct ranking: sort in descending order and slice exactly top_k items.
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    if top_k is None:
        return ranked_indices
    try:
        k = int(top_k)
    except Exception:
        k = 0
    k = max(0, k)
    return ranked_indices[:min(k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Original broken stepping logic (for reference, preserved as comment):
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    # Correct overlapping chunking with step = chunk_size - overlap.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    # Clamp overlap to a sensible range to ensure forward progress and intended overlap.
    overlap = max(0, min(int(overlap), max(0, chunk_size - 1)))
    step = chunk_size - overlap

    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original broken implementation (for reference, preserved as comment):
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }

    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Ensure numeric arrays
    arrays = [np.asarray(emb, dtype=float) for emb in embeddings]
    norms = [float(np.linalg.norm(arr)) for arr in arrays]

    # Variance across all raw embedding values (flattened), not just norms.
    all_values = np.concatenate([arr.ravel() for arr in arrays]) if arrays else np.array([])
    variance_values = float(np.var(all_values)) if all_values.size > 0 else 0.0

    # Also compute variance of norms; choose the larger positive signal to ensure robustness.
    variance_norms = float(np.var(norms)) if len(norms) > 0 else 0.0
    variance = variance_values if variance_values > 0.0 else variance_norms

    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p == 1) and (r == 1) for p, r in zip(predictions, relevance))
        fp = sum((p == 1) and (r == 0) for p, r in zip(predictions, relevance))
        fn = sum((p == 0) and (r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Original broken F1 computation (for reference, preserved as comment):
        # f1 = precision + recall  # This will be broken

        # Correct F1 score formula.
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
