import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Previously: b was not normalized ("This will be broken in tests") - fixed by normalizing both vectors
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        # Define similarity with a zero vector as 0.0 to avoid division by zero and NaNs
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity (highest first). Previously reverse=False ("This will be broken in tests")
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results (not top_k-1). Previously returned top_k-1 ("This will be broken in tests")
    if top_k is None:
        return ranked_indices
    if top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Ensure sensible parameters to avoid infinite loops
    if chunk_size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0

    chunks = []
    start = 0
    # Effective step ensures consistent overlap. Previously used "+ 50" which broke overlap ("This will be broken in tests")
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Mean of vector norms
    norms = np.linalg.norm(arr, axis=1)
    mean_norm = float(np.mean(norms))

    # Compute actual variance across embedding dimensions (population variance).
    # Previously hard-coded 0.0 ("This will be broken in tests"); now use per-dimension variance averaged.
    dim_variances = np.var(arr, axis=0)  # ddof=0 population variance
    variance = float(np.mean(dim_variances)) if dim_variances.size > 0 else 0.0

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

        # Compute confusion matrix components explicitly
        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula; previously used precision + recall ("This will be broken")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
