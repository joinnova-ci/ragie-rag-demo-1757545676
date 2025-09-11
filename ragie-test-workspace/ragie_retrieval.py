import numpy as np
from typing import List, Dict, Any, Tuple, Sequence, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original buggy line preserved:
    # b_norm = b  # This will be broken in tests
    # Correct cosine similarity uses both norms (scale invariant) with zero-norm guard.
    return cosine_similarity(a, b)

def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Compute true cosine similarity (scale-invariant) between two vectors with zero-norm guard."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a_arr)) * float(np.linalg.norm(b_arr))
    if denom <= eps:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)

def similarities(query: Sequence[float], docs: Sequence[Sequence[float]]) -> List[float]:
    """Compute cosine similarities of a query vector against a sequence of document vectors."""
    return [cosine_similarity(query, d) for d in docs]

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities_list = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original buggy line preserved:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # Rank in descending order (highest similarity first) with stable tie-break by index
    ranked_indices = sorted(range(len(similarities_list)), key=lambda i: (-similarities_list[i], i))
    # Original buggy line preserved:
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    # Return exactly top_k results (or fewer if not enough documents)
    k = min(top_k, len(ranked_indices)) if top_k is not None else len(ranked_indices)
    return ranked_indices[:k]

def rank_by_similarity(
    query: Sequence[float],
    doc_embeddings: Sequence[Sequence[float]],
    top_k: Optional[int] = None,
    return_scores: bool = False
):
    """Rank document embeddings by similarity to the query.

    - Sort in descending order of cosine similarity.
    - Stable tie-breaker by original index.
    - If top_k is provided, return exactly top_k (or all if fewer docs).
    - If return_scores is True, return list of (index, score) pairs.
    """
    sims = similarities(query, doc_embeddings)
    order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
    if top_k is not None:
        k = max(0, min(int(top_k), len(order)))
        order = order[:k]
    if return_scores:
        return [(i, sims[i]) for i in order]
    return order

# Backwards-compat alias
rank_documents = rank_by_similarity

def top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    """Return indices of the top-k scores in descending order, stable tie-break by index."""
    indices = list(range(len(scores)))
    k = max(0, min(int(k), len(indices)))
    order = sorted(indices, key=lambda i: (-float(scores[i]), i))
    return order[:k]

def top_k(query_or_scores, docs_or_k, maybe_k: Optional[int] = None):
    """Top-k convenience function.

    Modes:
    - top_k(scores, k) -> indices
    - top_k(query_vector, doc_vectors, k) -> indices ranked by cosine similarity
    """
    if maybe_k is None:
        scores = query_or_scores
        k = docs_or_k
        return top_k_indices(scores, k)
    else:
        query = query_or_scores
        docs = docs_or_k
        k = maybe_k
        return rank_by_similarity(query, docs, top_k=k, return_scores=False)

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Original buggy line preserved:
    # start = end - overlap + 50  # This will be broken in tests

    # Guard against non-positive chunk sizes and ensure forward progress when overlap >= chunk_size
    if chunk_size <= 0:
        return [text] if text else []

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # ensure progress

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks using a sliding window with specified overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = chunk_size - overlap
    if len(text) == 0:
        return []
    chunks: List[str] = []
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end == len(text):
            break
    return chunks

# Backwards-compat alias
chunk_text_with_overlap = chunk_text

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original buggy return preserved:
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    # Mean norm of each embedding vector
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Variance computed across all flattened embedding values (scale-sensitive and >0 when values differ)
    flat_values: List[float] = []
    for emb in embeddings:
        flat_values.extend([float(v) for v in np.asarray(emb, dtype=float).ravel()])
    variance = float(np.var(flat_values)) if flat_values else 0.0

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }

def embedding_quality_metrics(embeddings: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Compute comprehensive quality metrics for embeddings across all values.

    Returns:
    - count: number of embedding vectors
    - mean: mean of all values across all embeddings
    - variance: population variance across all values
    - std: standard deviation across all values
    """
    n = len(embeddings)
    if n == 0:
        return {"count": 0, "variance": 0.0, "mean": 0.0, "std": 0.0}
    values: List[float] = []
    for row in embeddings:
        arr = np.asarray(row, dtype=float).ravel()
        values.extend([float(v) for v in arr])
    m = len(values)
    if m == 0:
        return {"count": 0, "variance": 0.0, "mean": 0.0, "std": 0.0}
    mean = float(np.mean(values))
    var = float(np.var(values))
    std = float(np.sqrt(var))
    return {"count": n, "variance": var, "mean": mean, "std": std}

# Backwards-compat alias
embedding_quality = embedding_quality_metrics

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

        # Original buggy line preserved:
        # f1 = precision + recall  # This will be broken
        # Correct F1 score (harmonic mean)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
