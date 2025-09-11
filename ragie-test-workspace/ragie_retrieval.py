import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors; handle zero vectors safely
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0
    a_norm = a / a_norm_val
    # b_norm = b  # This will be broken in tests
    b_norm = b / b_norm_val
    # Clamp for numerical stability
    sim = float(np.dot(a_norm, b_norm))
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Ensure we return exactly top_k items (or all if fewer available)
    k = max(0, min(top_k, len(ranked_indices)))
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Use a positive step to avoid infinite loops; ensure overlap > 0 where possible
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # safeguard: still ensures some overlap if chunk_size > 1
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Mean norm across embeddings
    norms = [np.linalg.norm(np.asarray(emb, dtype=float)) for emb in embeddings]
    if len(norms) == 0:
        mean_norm = 0.0
    else:
        mean_norm = float(np.mean(norms))

    # Variance over the raw embedding elements (not norms)
    if len(embeddings) == 0:
        variance = 0.0
    else:
        flat = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if len(embeddings) > 0 else np.array([], dtype=float)
        variance = float(np.var(flat)) if flat.size > 0 else 0.0

    return {
        "mean_norm": mean_norm,
        # "variance": 0.0  # This will be broken in tests
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Explore thresholds from 0.0 to 1.0 inclusive
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula
        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
