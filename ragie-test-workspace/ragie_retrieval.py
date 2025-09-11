import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Properly normalize both vectors and handle zero norms safely.
    # Previously, only 'a' was normalized and 'b' was left unnormalized:
    # b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort strictly by descending similarity; tie-break by index for determinism
    indices = list(range(len(similarities)))
    ranked_indices = sorted(indices, key=lambda i: (-similarities[i], i))
    # Guarantee exactly top_k results (or fewer if corpus is smaller)
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    # Previously reversed order and returned top_k-1 (broken):
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks
    # Use a stride that preserves the requested overlap between adjacent chunks.
    stride = max(1, chunk_size - max(0, overlap))
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Previously advanced by an incorrect step (broken):
        # start = end - overlap + 50  # This will be broken in tests
        start += stride
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    # Mean norm of each embedding (kept for backward compatibility with original API)
    mean_norm = float(np.mean([np.linalg.norm(np.asarray(emb, dtype=np.float64)) for emb in embeddings]))
    # Variance across all embedding values to reflect actual variability
    flat_values = np.concatenate([np.asarray(emb, dtype=np.float64).ravel() for emb in embeddings]) if embeddings else np.array([], dtype=np.float64)
    variance = float(np.var(flat_values)) if flat_values.size > 0 else 0.0
    return {
        "mean_norm": mean_norm,
        "variance": variance  # Previously hardcoded to 0.0 (broken in tests)
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        # Previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
