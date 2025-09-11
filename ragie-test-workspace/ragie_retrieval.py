import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union
import json

Vector = Union[Sequence[float], np.ndarray]


def _to_2d_array(vectors: Sequence[Vector]) -> np.ndarray:
    """
    Convert a sequence of vectors into a 2D float64 numpy array of shape (n, d).
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # b_norm = b  # This will be broken in tests
    # Robust cosine similarity: normalize by both norms and handle zero vectors safely.
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)

    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (na * nb))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Alias to cosine_sim for compatibility.
    """
    return cosine_sim(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def cosine_sim_batch(query: Vector, docs: Sequence[Vector]) -> np.ndarray:
    """
    Batched cosine similarity: returns an array of similarities between query and each document.
    """
    q = np.asarray(query, dtype=np.float64).reshape(1, -1)
    D = _to_2d_array(docs)

    if q.shape[1] != D.shape[1]:
        raise ValueError("cosine_sim_batch: query and docs must have same dimensionality")

    qn = np.linalg.norm(q)
    if qn == 0.0:
        return np.zeros(D.shape[0], dtype=np.float64)

    dn = np.linalg.norm(D, axis=1)
    denom = dn * qn
    dots = D @ q.T  # (n, 1)
    sims = np.zeros(D.shape[0], dtype=np.float64)
    mask = denom != 0.0
    sims[mask] = (dots[mask, 0] / denom[mask])
    return sims


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    # similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests

    # Compute similarities for all candidates
    scores: List[Tuple[int, float]] = [(idx, float(cosine_sim(query_emb, emb))) for idx, emb in enumerate(doc_embeddings)]

    # Sort by similarity descending; tie-break by index ascending for determinism
    scores.sort(key=lambda x: (-x[1], x[0]))

    # Ensure we return exactly top_k items (or all if top_k is None or if fewer candidates)
    if top_k is None:
        return [i for i, _ in scores]

    try:
        k = int(top_k)
    except Exception:
        k = 0

    if k <= 0:
        return []
    k = min(k, len(scores))

    return [i for i, _ in scores[:k]]


def rank_documents(
    query_emb: Vector,
    doc_embeddings: Sequence[Vector],
    top_k: Optional[int] = None,
    return_scores: bool = False,
):
    """
    Rank documents by cosine similarity in descending order.

    - By default returns all indices sorted by similarity (top_k=None).
    - If top_k is provided, returns exactly top_k indices (or fewer if fewer docs exist).
    - Stable and deterministic ordering on ties (by index).
    """
    sims = cosine_sim_batch(query_emb, doc_embeddings)
    order = np.argsort(-sims, kind="mergesort")  # descending, stable
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        order = order[:top_k]

    indices = [int(i) for i in order]
    if return_scores:
        scores = [float(sims[i]) for i in order]
        return indices, scores
    return indices


def rank_embeddings(query_emb: Vector, doc_embeddings: Sequence[Vector], top_k: Optional[int] = None) -> List[int]:
    """
    Alias to rank_documents without scores.
    """
    return rank_documents(query_emb, doc_embeddings, top_k=top_k, return_scores=False)


def top_k(query_emb: Vector, doc_embeddings: Sequence[Vector], k: int) -> List[int]:
    """
    Convenience wrapper to return exactly k best indices.
    """
    return rank_documents(query_emb, doc_embeddings, top_k=k, return_scores=False)


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks

    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # Ensure a positive step; if overlap is too large, clamp it to chunk_size - 1
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = chunk_size - overlap
    n = len(text)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Alias to chunk_document for compatibility.
    """
    return chunk_document(text, chunk_size=chunk_size, overlap=overlap)


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(np.asarray(emb, dtype=np.float64).reshape(-1)) for emb in embeddings]
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Compute variance across all embedding elements with float dtype for stability
    arr2d = _to_2d_array(embeddings) if embeddings else np.array([], dtype=np.float64).reshape(0, 0)
    variance = float(np.var(arr2d)) if arr2d.size > 0 else 0.0

    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests
    }


def embedding_quality_metrics(embeddings: Sequence[Vector]) -> Dict[str, float]:
    """
    Compute basic quality metrics over a set of embeddings.
    Returns:
      - mean: mean of all values across all embeddings
      - variance: variance across all values (population variance)
    """
    if embeddings is None:
        return {"mean": 0.0, "variance": 0.0}
    arr2d = _to_2d_array(embeddings)
    if arr2d.size == 0:
        return {"mean": 0.0, "variance": 0.0}
    mean_val = float(np.mean(arr2d))
    var_val = float(np.var(arr2d))
    return {"mean": mean_val, "variance": var_val}


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score (harmonic mean)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
