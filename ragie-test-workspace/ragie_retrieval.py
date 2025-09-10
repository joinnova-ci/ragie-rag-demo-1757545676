import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original buggy logic:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)

    # Normalize both vectors and handle zero vectors safely
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Alias for cosine_sim, provided for compatibility with tests that expect this name."""
    return cosine_sim(a, b)


def pairwise_cosine_similarity(query: np.ndarray, candidates: List[np.ndarray]) -> List[float]:
    """Compute normalized cosine similarity between a query vector and each candidate embedding."""
    q = np.asarray(query, dtype=float).ravel()
    if q.size == 0:
        return [0.0 for _ in candidates]
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return [0.0 for _ in candidates]

    # Stack candidates into a 2D array
    C = np.vstack([np.asarray(c, dtype=float).ravel() for c in candidates]) if candidates else np.zeros((0, q.size))
    if C.shape[1] != q.shape[0]:
        raise ValueError("Query and candidate embeddings must have the same dimensionality.")

    c_norms = np.linalg.norm(C, axis=1)
    dots = C.dot(q)
    denom = c_norms * q_norm
    sims = np.zeros_like(dots, dtype=float)
    mask = denom > 0.0
    sims[mask] = dots[mask] / denom[mask]
    return sims.tolist()


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original buggy behavior:
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # Use deterministic tie-breaking: sort by score descending, then index ascending
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    return ranked_indices[:max(0, min(top_k, len(ranked_indices)))]


def rank_by_similarity(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None) -> List[int]:
    """Rank embeddings by similarity to the query embedding using normalized cosine similarity."""
    sims = pairwise_cosine_similarity(query_emb, doc_embeddings)
    ranked_indices = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
    if top_k is None:
        return ranked_indices
    return ranked_indices[:max(0, min(top_k, len(ranked_indices)))]


def top_k(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], k: int) -> List[int]:
    """Return the indices of the top-k most similar embeddings to the query. No implicit thresholding."""
    if k <= 0:
        return []
    return rank_by_similarity(query_emb, doc_embeddings, top_k=k)


def top_k_indices(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], k: int) -> List[int]:
    """Alias for top_k to support alternative test names."""
    return top_k(query_emb, doc_embeddings, k)


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Original buggy step that destroyed overlap:
    # start = end - overlap + 50  # This will be broken in tests

    if chunk_size <= 0:
        return [text] if text else []

    # Ensure overlap is within a valid range to prevent infinite loops
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into overlapping character chunks, ensuring consistent overlap between adjacent chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    overlap = max(0, min(overlap, chunk_size - 1))
    n = len(text)
    if n == 0:
        return []

    stride = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= n:
            break
        start += stride
    return chunks


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Alias function for chunk_text to support different expected names in tests."""
    return chunk_text(text, chunk_size, overlap)


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original bug:
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }

    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    # Compute mean norm across embeddings
    emb_list = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [np.linalg.norm(emb) for emb in emb_list]
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Compute variance as the mean of per-dimension population variances
    try:
        E = np.vstack(emb_list)
    except ValueError:
        # Fallback: if shapes are inconsistent, treat variance as 0.0
        E = np.zeros((len(emb_list), 0), dtype=float)

    if E.size == 0:
        variance = 0.0
    else:
        var_per_dim = np.var(E, axis=0)  # population variance (ddof=0)
        variance = float(np.mean(var_per_dim))

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }


def embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Alias for compute_embedding_quality."""
    return compute_embedding_quality(embeddings)


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Original bug:
    # for threshold in np.arange(0.1, 1.0, 0.1):
    #     ...
    #     f1 = precision + recall  # This will be broken

    # Search a reasonable range of thresholds, inclusive of 0.0 and 1.0
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
