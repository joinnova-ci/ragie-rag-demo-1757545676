import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    # If dimensions mismatch, compare over the common length to avoid errors
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    denom = a_norm * b_norm
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    val = float(np.dot(a, b) / denom)
    # Clamp to account for tiny numerical drift outside [-1, 1]
    if not np.isfinite(val):
        return 0.0
    return max(min(val, 1.0), -1.0)

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = []
    for doc_emb in doc_embeddings:
        sim = cosine_sim(query_emb, doc_emb)
        # Treat non-finite similarities as the lowest possible to push them to the end
        if not np.isfinite(sim):
            sim = float("-inf")
        similarities.append(sim)

    # Sort in descending order of similarity, tie-breaking by index to ensure deterministic order
    sims = np.asarray(similarities, dtype=np.float64)
    indices = np.arange(len(sims))
    # np.lexsort uses the last key as the primary key; we want to sort by -sims (desc), then by index (asc)
    order = np.lexsort((indices, -sims))

    if top_k is None:
        k = len(doc_embeddings)
    else:
        if top_k <= 0:
            return []
        k = min(int(top_k), len(doc_embeddings))

    return order[:k].tolist()

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        # Ensure positive step to avoid infinite loops
        overlap = chunk_size - 1

    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(np.asarray(emb, dtype=float)) for emb in embeddings]
    # Compute variance across norms of embeddings (consistency across vectors)
    if norms:
        mean_norm = float(np.mean(norms))
        variance = float(np.var(norms)) if len(norms) > 1 else 0.0
    else:
        mean_norm = 0.0
        variance = 0.0
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
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
