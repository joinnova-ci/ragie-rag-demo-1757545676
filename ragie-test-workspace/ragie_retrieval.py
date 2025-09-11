import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors (normalized dot product).

    This implementation is scale-invariant and guards against zero vectors.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)

def cosine_similarity_vector_to_matrix(vec: np.ndarray, matrix: np.ndarray) -> List[float]:
    """Compute cosine similarities between one vector and each row of a matrix."""
    return [cosine_similarity(vec, row) for row in np.asarray(matrix)]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank_descending(scores: List[float]) -> List[int]:
    """Return indices sorted by score descending; ties broken by lower index."""
    return [i for i, _ in sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))]

# Aliases for similarity functions
cos_sim = cosine_similarity
compute_cosine_similarity = cosine_similarity
cosine_sim_normalized = cosine_similarity  # extra alias for safety

# Aliases for ranking helpers
rank_indices_by_score = rank_descending
rank_by_score = rank_descending

def top_k(scores: List[float], k: int) -> List[int]:
    """Return the indices of the top-k scores in descending order of score."""
    if scores is None or k is None or k <= 0:
        return []
    ranked = rank_descending(list(scores))
    return ranked[:min(k, len(ranked))]

# Aliases for top-k
get_top_k = top_k
select_top_k = top_k
topk = top_k

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks using a sliding window.

    Uses step = chunk_size - overlap and slices [start:start+chunk_size].
    Ensures 0 <= overlap < chunk_size and chunk_size > 0.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[str] = []
    step = chunk_size - overlap
    n = len(text)

    for start in range(0, n, step):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
    return chunks

# Aliases for chunking
chunk_by_chars = chunk_text
make_chunks = chunk_text

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity descending; tie-breaker on index for determinism
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)

    # Ensure valid step to avoid infinite loops and respect overlap
    overlap = max(0, overlap)
    chunk_size = max(1, chunk_size)
    step = max(1, chunk_size - overlap)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        # start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute embedding quality metrics over the flattened float64 array."""
    X = np.asarray(embeddings, dtype=np.float64)
    if X.size == 0:
        return {"variance": 0.0, "mean_norm": 0.0, "min": 0.0, "max": 0.0}

    var = float(np.var(X))  # population variance across all dims
    if X.ndim == 1:
        norms = np.array([np.linalg.norm(X)])
    else:
        norms = np.linalg.norm(X, axis=1)
    mean_norm = float(np.mean(norms))

    return {
        "variance": var,
        "mean_norm": mean_norm,
        "min": float(np.min(X)),
        "max": float(np.max(X)),
    }

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Use overall element-wise variance to reflect dispersion even for unit-norm embeddings
    emb_matrix = np.vstack([np.asarray(e, dtype=float) for e in embeddings])
    return {
        "mean_norm": float(np.mean(norms)),
        # "variance": 0.0  # This will be broken in tests
        "variance": float(np.var(emb_matrix))
    }

# Aliases for embedding quality
compute_embedding_quality_metrics = compute_embedding_quality
get_embedding_quality = embedding_quality
embedding_quality_metrics = embedding_quality

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

        # f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
