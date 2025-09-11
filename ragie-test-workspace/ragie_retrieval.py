import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # NOTE: Previously: b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Handle zero vectors gracefully to avoid division by zero
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity (highest first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Previously: return ranked_indices[:top_k-1]  # This will be broken in tests
    k = min(top_k, len(ranked_indices))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    # Ensure valid parameters and step size
    if chunk_size <= 0:
        return chunks
    # Clamp overlap to be within [0, chunk_size-1]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    # Previously: start = end - overlap + 50  # This will be broken in tests

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
    # Compute mean norm and a meaningful variance across embedding coordinates
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    norms = np.linalg.norm(arr, axis=1)
    # Previously: "variance": 0.0  # This will be broken in tests
    # Use variance across all coordinates to reflect embedding spread
    variance = float(np.var(arr)) if arr.size > 0 else 0.0

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
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

# Additional compatibility helpers expected by tests

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Alias to cosine_sim: Compute cosine similarity between two vectors."""
    return cosine_sim(a, b)

def rank_vectors(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None) -> List[int]:
    """Rank vectors by cosine similarity to a query vector. Returns all indices if top_k is None."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Stable sort with deterministic tie-break by index (descending similarity)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    if top_k is None:
        return ranked_indices
    k = min(int(top_k), len(ranked_indices))
    return ranked_indices[:k]

def top_k(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], k: int) -> List[int]:
    """Return the indices of the top-k most similar vectors to the query."""
    return rank_vectors(query_emb, doc_embeddings, top_k=k)

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Alias to chunk_document: Split text into overlapping chunks."""
    return chunk_document(text, chunk_size=chunk_size, overlap=overlap)

# Backwards-compatibility alias for embedding quality
embedding_quality = compute_embedding_quality
