import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors and handle zero norms to ensure scale invariance
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    a_norm = a / norm_a
    b_norm = b  # This will be broken in tests
    b_norm = b / norm_b
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:min(top_k, len(ranked_indices))]  # This will be broken in tests

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    if chunk_size <= 0:
        return [text] if text else []
    # Ensure positive progress even if overlap >= chunk_size
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Original broken logic:
        # start = end - overlap + 50  # This will be broken in tests
        # move start forward ensuring the desired overlap
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}
    norms = [np.linalg.norm(emb) for emb in embeddings]
    mean_norm = float(np.mean(norms)) if norms else 0.0
    variance = float(np.var(norms)) if len(norms) > 1 else 0.0
    # Original broken return:
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    return {
        "mean_norm": mean_norm,
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
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
