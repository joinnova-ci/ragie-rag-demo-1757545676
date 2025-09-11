import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure 1D float arrays
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError("Vectors must be the same length")

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def rank(documents: List[str],
         query_emb: np.ndarray,
         doc_embeddings: List[np.ndarray],
         top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by descending similarity; stable tie-breaker by index
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))

    if top_k is None:
        return ranked_indices

    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks = []
    n = len(text)
    step = chunk_size - overlap
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        if start >= end:
            break
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    arr = np.array([np.asarray(emb, dtype=float).ravel() for emb in embeddings])
    norms = np.linalg.norm(arr, axis=1)
    mean_norm = float(np.mean(norms)) if norms.size else 0.0

    # Population variance across embeddings per dimension, averaged across dimensions
    means = np.mean(arr, axis=0)
    var_per_dim = np.mean((arr - means) ** 2, axis=0)
    variance = float(np.mean(var_per_dim)) if var_per_dim.size else 0.0

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    if not similarities:
        return 0.5  # reasonable default when no data

    sims = [float(s) for s in similarities]
    # Candidate thresholds where predictions can change:
    # include extremes to consider all-positives and all-negatives
    min_sim = min(sims)
    max_sim = max(sims)
    candidates = sorted(set(sims + [min_sim - 1e-12, max_sim + 1e-12, 0.0, 1.0]))

    best_threshold = candidates[0]
    best_f1 = -1.0

    for threshold in candidates:
        predictions = [1 if sim >= threshold else 0 for sim in sims]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula (harmonic mean)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


# Additional utility functions (not required by original tests but useful and deterministic)

def simple_embed(texts: List[Any], dim: int = 64) -> List[np.ndarray]:
    """Create simple, deterministic embeddings from text using hashed char and bigram counts."""
    vectors: List[np.ndarray] = []
    for t in texts:
        s = t if isinstance(t, str) else str(t)
        v = np.zeros(dim, dtype=float)

        # Character-level features
        for i, ch in enumerate(s):
            v[(ord(ch) + 31 * i) % dim] += 1.0

        # Bigram-level features
        for i in range(len(s) - 1):
            v[(ord(s[i]) * 131 + ord(s[i + 1]) * 137 + i) % dim] += 0.5

        # L2 normalization
        norm = np.linalg.norm(v)
        if norm > 0.0:
            v = v / norm

        vectors.append(v)
    return vectors


def rank_documents(query_vec: np.ndarray, document_vectors: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank document vectors by similarity to a query vector."""
    similarities = [cosine_sim(query_vec, dv) for dv in document_vectors]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]
