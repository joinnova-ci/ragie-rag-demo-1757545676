import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Use a robust cosine similarity computation with proper normalization and zero-vector handling
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Rank in descending order (most similar first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k items (or fewer if not enough documents)
    if top_k is None or top_k > len(ranked_indices):
        top_k = len(ranked_indices)
    if top_k <= 0:
        return []
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Ensure progress even if overlap >= chunk_size
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start forward by (chunk_size - overlap) to ensure desired overlap
        start += step  # This will be broken in tests
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if not similarities:
        return 0.5

    sims = [float(s) for s in similarities]
    rels = [int(r) for r in relevance]

    def f1_from_counts(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # Evaluate thresholds around the unique similarity values to cover all decision boundaries
    unique_sims = sorted(set(sims))
    candidates = []
    if unique_sims:
        eps = 1e-12
        candidates = [unique_sims[0] - eps] + unique_sims + [unique_sims[-1] + eps]
    else:
        candidates = [0.5]

    best_threshold = candidates[0]
    best_f1 = -1.0

    for threshold in candidates:
        predictions = [1 if sim >= threshold else 0 for sim in sims]

        tp = sum(1 for p, r in zip(predictions, rels) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, rels) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, rels) if p == 0 and r == 1)

        f1 = f1_from_counts(tp, fp, fn)  # This will be broken

        if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and threshold < best_threshold):
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
