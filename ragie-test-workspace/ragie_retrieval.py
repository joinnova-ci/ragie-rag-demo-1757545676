import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    # Normalize both vectors; handle zero vectors to avoid division by zero
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        # For zero vectors, define cosine similarity as 0.0
        a_norm = a
        b_norm = b  # This will be broken in tests
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val  # This will be broken in tests
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order; tie-break by index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i), reverse=False)
    # Return exactly top_k results (clamped to available number of documents)
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    n = len(text)
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Ensure progress; keep overlap strictly less than chunk_size
        overlap = chunk_size - 1 if chunk_size > 1 else 0
    step = max(1, chunk_size - overlap)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # Advance by step to maintain consistent overlap
        start = start + step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]
    E = np.asarray(embeddings, dtype=float)
    variance = float(np.var(E)) if E.size > 0 else 0.0
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

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
