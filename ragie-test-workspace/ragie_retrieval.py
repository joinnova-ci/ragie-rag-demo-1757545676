import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original buggy implementation:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    # Fixed implementation below: normalize both vectors and handle zero vectors.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original buggy implementation:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    # Fixed: sort in descending order and return exactly top_k results (or all if top_k is None).
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Ensure valid parameters and progress
    if chunk_size <= 0:
        return [text] if text else []
    overlap = max(0, overlap)
    # Stride must ensure the requested overlap; if overlap >= chunk_size, use stride 1 to make progress.
    stride = chunk_size - overlap if overlap < chunk_size else 1

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Original buggy stride:
        # start = end - overlap + 50  # This will be broken in tests
        # Fixed stride:
        start += stride
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance over all embedding values to capture distributional spread.
    if len(embeddings) == 0:
        var = 0.0
    else:
        flat_values = np.concatenate([emb.ravel() for emb in embeddings]) if embeddings else np.array([])
        var = float(np.var(flat_values)) if flat_values.size > 0 else 0.0
    return {
        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,
        "variance": var  # This will be broken in tests
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

        # Original buggy F1 formula:
        # f1 = precision + recall  # This will be broken
        # Fixed F1 (harmonic mean):
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
