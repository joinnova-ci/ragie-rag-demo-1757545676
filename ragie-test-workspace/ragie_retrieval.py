import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Previous buggy version:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    denom = a_norm * b_norm
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return 0.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Previous buggy version:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # Sort by descending similarity; deterministic tie-breaker by index
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))

    # Return exactly min(top_k, n_docs) results
    k = int(top_k) if top_k is not None else len(ranked_indices)
    if k < 0:
        k = 0
    if k > len(ranked_indices):
        k = len(ranked_indices)
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Previous buggy stepping logic:
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # ensure progress
        overlap = chunk_size - 1

    step = chunk_size - overlap
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Previous buggy version:
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    if embeddings is None:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    arr = np.asarray(embeddings, dtype=float)
    if arr.size == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }

    # Compute statistics based on the norms of each embedding to reflect scale differences
    if arr.ndim == 1:
        norms = np.array([np.linalg.norm(arr)], dtype=float)
    else:
        norms = np.linalg.norm(arr, axis=1)

    mean_norm = float(np.mean(norms))
    variance = float(np.var(norms))  # variance of norms (not raw elements)
    return {
        "mean_norm": mean_norm,
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

        # Correct F1 score (harmonic mean), replacing the buggy sum:
        # f1 = precision + recall  # This will be broken
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
