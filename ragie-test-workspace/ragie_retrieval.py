import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Original buggy approach only normalized 'a' and not 'b'.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original incorrect implementation:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    k = max(0, min(top_k, len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Ensure valid parameters to avoid infinite loops or negative steps
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        # Guarantee forward progress; keep behavior consistent but safe
        overlap = chunk_size - 1

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Original incorrect step:
        # start = end - overlap + 50  # This will be broken in tests
        start = end - overlap  # move forward by chunk_size - overlap
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across embedding coordinates rather than just norms
    # Original incorrect implementation returned a constant variance:
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    try:
        matrix = np.vstack([np.asarray(emb, dtype=float) for emb in embeddings])
        var = float(np.var(matrix))
    except ValueError:
        # Handle ragged embeddings by computing variance over flattened concatenation
        flat = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if embeddings else np.array([])
        var = float(np.var(flat)) if flat.size > 0 else 0.0

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": var
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Original incorrect F1 formulation:
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
