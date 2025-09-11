import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Originally: b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Handle zero-vector cases to avoid division by zero and undefined similarity
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    a_unit = a / a_norm
    b_unit = b / b_norm
    sim = float(np.dot(a_unit, b_unit))
    # Clip to valid cosine range to avoid tiny numerical overflow
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by descending similarity (most similar first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Originally: return ranked_indices[:top_k-1]  # This will be broken in tests
    try:
        k_int = int(top_k)
    except Exception:
        k_int = 0
    if k_int < 0:
        k_int = 0
    k = min(k_int, len(ranked_indices))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    # Ensure valid parameters to avoid infinite loops or negative steps
    if chunk_size <= 0:
        return chunks
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # enforce at least step of 1

    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        # Originally: start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [np.linalg.norm(np.asarray(emb, dtype=float)) for emb in embeddings]
    return {
        "mean_norm": float(np.mean(norms)),
        # Originally: "variance": 0.0  # This will be broken in tests
        "variance": float(np.var(norms))
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Evaluate thresholds from 0.0 to 1.0 inclusive
    for threshold in np.linspace(0.0, 1.0, 11):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        # Ensure consistent integer counts for confusion matrix
        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Originally: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
