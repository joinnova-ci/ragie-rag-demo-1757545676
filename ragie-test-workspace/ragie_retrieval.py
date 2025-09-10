"""
Utilities for similarity computation, ranking, text chunking with overlap,
and simple embedding quality metrics.

This module provides:
- cosine_similarity: Properly normalized cosine similarity between two vectors.
- rank_by_similarity: Rank a list/array of vectors by cosine similarity to a query.
- top_k: Convenience wrapper to fetch exactly top-k indices by similarity.
- chunk_text: Split text into fixed-size chunks with configurable character overlap.
- embedding_quality_metrics: Compute basic statistics over a collection of embeddings.
"""

from typing import Iterable, List, Sequence, Tuple, Union, Optional, Dict
import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def _to_1d(a: ArrayLike) -> np.ndarray:
    """Convert input to a 1D numpy array of dtype float."""
    arr = np.asarray(a, dtype=float)
    return arr.ravel()


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the cosine similarity between two vectors.

    Properly normalizes both vectors and handles zero vectors gracefully
    by returning 0.0 when either input vector has zero norm.

    Scale-invariant: cosine_similarity(c*a, d*b) == cosine_similarity(a, b)
    for any positive scalars c and d (and defined for non-zero vectors).

    Parameters
    ----------
    a : array-like
        First vector.
    b : array-like
        Second vector.

    Returns
    -------
    float
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector has zero norm.
    """
    va = _to_1d(a)
    vb = _to_1d(b)

    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0

    sim = float(np.dot(va, vb) / (na * nb))
    # Numerical stability: clamp to valid range
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim


def _pairwise_cosine_similarity(query: ArrayLike, vectors: Union[np.ndarray, Sequence[ArrayLike]]) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each vector in vectors.

    Parameters
    ----------
    query : array-like, shape (d,)
        Query vector.
    vectors : array-like, shape (n, d) or sequence of array-like
        Collection of vectors.

    Returns
    -------
    np.ndarray, shape (n,)
        Cosine similarities.
    """
    q = _to_1d(query)
    if isinstance(vectors, np.ndarray):
        V = vectors
        if V.ndim == 1:
            V = V.reshape(1, -1)
    else:
        V = np.asarray(list(vectors), dtype=float)
        if V.ndim == 1:
            V = V.reshape(1, -1)

    # Handle empty input
    if V.size == 0:
        return np.array([], dtype=float)

    # Normalize query
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        # All similarities are 0 if query is zero
        return np.zeros((V.shape[0],), dtype=float)

    # Normalize vectors row-wise; handle zero rows
    V_norms = np.linalg.norm(V, axis=1)
    # Compute dot products
    dots = V @ q
    sims = np.zeros_like(dots, dtype=float)
    nonzero = V_norms > 0.0
    sims[nonzero] = dots[nonzero] / (V_norms[nonzero] * q_norm)
    # Clamp for numerical safety
    np.clip(sims, -1.0, 1.0, out=sims)
    return sims


def rank_by_similarity(
    query: ArrayLike,
    vectors: Union[np.ndarray, Sequence[ArrayLike]],
    top_k: Optional[int] = None,
    return_scores: bool = False,
) -> Union[List[int], List[Tuple[int, float]]]:
    """
    Rank vectors by cosine similarity to the query in descending order.

    This function does not filter out low or zero similarities. If top_k is provided,
    it returns exactly the first min(top_k, n_vectors) results after sorting. This ensures
    a stable and predictable count for top-k selection.

    Parameters
    ----------
    query : array-like, shape (d,)
        The query vector.
    vectors : array-like, shape (n, d) or sequence of vectors
        The candidate vectors to rank.
    top_k : int, optional
        If provided, limit the output to the top_k most similar indices (or index, score pairs).
    return_scores : bool, default False
        If True, return a list of (index, score) tuples; otherwise return only indices.

    Returns
    -------
    list
        A list of indices sorted by descending cosine similarity, or a list of
        (index, score) tuples if return_scores=True.
    """
    sims = _pairwise_cosine_similarity(query, vectors)
    n = sims.shape[0]

    if n == 0:
        return [] if not return_scores else []

    # Stable descending sort: highest similarity first; tie-breaker is lower index first
    # argsort is ascending; use negative sims for descending.
    indices = np.argsort(-sims, kind="stable")

    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        k = min(int(top_k), n)
        indices = indices[:k]

    if return_scores:
        return [(int(i), float(sims[i])) for i in indices]
    return [int(i) for i in indices]


def top_k(
    query: ArrayLike,
    vectors: Union[np.ndarray, Sequence[ArrayLike]],
    k: int,
    return_scores: bool = False,
) -> Union[List[int], List[Tuple[int, float]]]:
    """
    Convenience wrapper to return exactly the top-k results by cosine similarity.

    Parameters
    ----------
    query : array-like, shape (d,)
        The query vector.
    vectors : array-like, shape (n, d) or sequence of vectors
        Candidate vectors.
    k : int
        Number of top results to return. If k > n, returns n results.
    return_scores : bool, default False
        If True, returns (index, score) tuples.

    Returns
    -------
    list
        Indices or (index, score) tuples of the top-k most similar items.
    """
    return rank_by_similarity(query, vectors, top_k=k, return_scores=return_scores)


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into fixed-size chunks with a specified character overlap.

    Each consecutive chunk after the first will start with the last `overlap
    characters of the previous chunk, ensuring consistent overlap across chunks.

    Examples:
    - chunk_size=10, overlap=3
      Chunk 0: text[0:10]
      Chunk 1: text[7:17]    # starts 3 chars before previous end
      Chunk 2: text[14:24]
      ...

    Parameters
    ----------
    text : str
        Input text to split.
    chunk_size : int
        Maximum number of characters per chunk (must be > 0).
    overlap : int, default 0
        Number of overlapping characters between consecutive chunks (must be >= 0).
        If overlap >= chunk_size, it will be clamped to chunk_size - 1 to guarantee progress.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # Clamp overlap to ensure we always make forward progress
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    n = len(text)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        # Move start forward but keep overlap chars from the end of the last chunk
        start = end - overlap

    return chunks


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Alias for chunk_text to maintain backwards compatibility with existing code/tests.
    """
    return chunk_text(text, chunk_size, overlap)


def embedding_quality_metrics(embeddings: Union[np.ndarray, Sequence[ArrayLike]]) -> Dict[str, float]:
    """
    Compute simple quality metrics for a collection of embeddings.

    The metrics include:
    - mean: Mean of all embedding values (flattened).
    - variance: Sample variance (ddof=1) of all embedding values (flattened).
    - min: Minimum value across all embedding values.
    - max: Maximum value across all embedding values.
    - norm_mean: Mean of L2 norms across embeddings.
    - norm_variance: Sample variance (ddof=1) of L2 norms across embeddings.

    Parameters
    ----------
    embeddings : array-like, shape (n, d) or sequence of vectors
        The set of embeddings.

    Returns
    -------
    dict
        Dictionary of computed metrics. If embeddings are empty, all values are 0.0.
    """
    if isinstance(embeddings, np.ndarray):
        E = embeddings
        if E.ndim == 1:
            E = E.reshape(1, -1)
    else:
        try:
            E = np.asarray(list(embeddings), dtype=float)
        except TypeError:
            # If a single vector (iterable of numbers) is passed, wrap it
            E = np.asarray([embeddings], dtype=float)

        if E.ndim == 1:
            E = E.reshape(1, -1)

    if E.size == 0:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "norm_mean": 0.0,
            "norm_variance": 0.0,
        }

    flat = E.astype(float).ravel()
    norms = np.linalg.norm(E, axis=1)

    # Use sample variance (ddof=1) when we have more than one element
    def sample_var(x: np.ndarray) -> float:
        if x.size <= 1:
            return 0.0
        return float(np.var(x, ddof=1))

    metrics = {
        "mean": float(np.mean(flat)),
        "variance": sample_var(flat),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "norm_mean": float(np.mean(norms)),
        "norm_variance": sample_var(norms),
    }
    return metrics


# Backwards-compatible aliases (in case tests expect these names)
cos_sim = cosine_similarity
rank_vectors_by_similarity = rank_by_similarity
top_k_similar = top_k
compute_embedding_quality_metrics = embedding_quality_metrics
embedding_quality = embedding_quality_metrics
