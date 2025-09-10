"""
Utilities for text chunking, similarity computation, ranking, and simple
embedding quality metrics used in retrieval workflows.

This module provides:
- Cosine similarity that is properly normalized and scale invariant.
- Deterministic ranking helpers that sort scores in descending order.
- Top-k selection that returns exactly k indices (when k > 0).
- Robust text chunking with fixed character overlap between adjacent chunks.
- Simple embedding quality metrics including variance.

The implementations are defensive against common edge cases such as empty
inputs, zero vectors, and invalid parameters, and aim to be deterministic
for reproducible tests.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Compute the cosine similarity between two vectors.

    The cosine similarity is defined as:
        cos_sim(a, b) = (a · b) / (||a|| * ||b||)

    Properties:
    - Returns values in [-1.0, 1.0].
    - Scale invariant: scaling either vector by any positive scalar
      does not change the result.
    - If either vector has zero magnitude, returns 0.0.

    Parameters
    ----------
    vec_a : Sequence[float]
        First vector.
    vec_b : Sequence[float]
        Second vector.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    # Support different sequence lengths by operating on the overlapping portion.
    # In typical use they should be the same length, but being lenient prevents crashes.
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        a = float(vec_a[i])
        b = float(vec_b[i])
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        # One or both are zero vectors -> undefined cosine, return neutral 0.0
        return 0.0

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    # Guard against extremely small denominators due to numerical underflow
    if denom == 0.0:
        return 0.0

    sim = dot / denom
    # Numerical rounding can lead to tiny excursions beyond [-1, 1]
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim


# Common alias used in tests and other modules
cosine_sim = cosine_similarity


# ---------------------------------------------------------------------------
# Text chunking with overlap
# ---------------------------------------------------------------------------

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of maximum `chunk_size` characters, ensuring that
    adjacent chunks have exactly `overlap` characters in common whenever possible.

    The step between chunk starts is: step = max(1, chunk_size - overlap)

    Notes:
    - If `text` is empty, returns [].
    - If `chunk_size` <= 0, returns [] (nothing to chunk).
    - If `overlap` < 0, it is treated as 0.
    - If `overlap` >= chunk_size, it is clamped to chunk_size - 1 to ensure progress.
    - Overlap is measured in characters.
    - The last chunk may be shorter than `chunk_size`, and thus may have a smaller
      effective overlap with the previous chunk.

    Parameters
    ----------
    text : str
        Input text to chunk.
    chunk_size : int
        Maximum size of each chunk in characters.
    overlap : int
        Number of overlapping characters between adjacent chunks.

    Returns
    -------
    List[str]
        A list of chunk strings.
    """
    if not text:
        return []
    if chunk_size is None or chunk_size <= 0:
        return []

    if overlap is None:
        overlap = 0
    # Clamp overlap to a valid range
    overlap = max(0, min(int(overlap), max(0, chunk_size - 1)))

    step = chunk_size - overlap
    # Safety: ensure forward progress even if overlap == chunk_size - 1
    if step <= 0:
        step = 1

    chunks: List[str] = []
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= len(text):
            break

    return chunks


# Common aliases
chunk_text = chunk_text_with_overlap
chunk_document = chunk_text_with_overlap
chunk_with_overlap = chunk_text_with_overlap


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def rank_indices_by_scores(scores: Sequence[float]) -> List[int]:
    """
    Return indices that would sort the scores in descending order.

    Ties are broken deterministically by index (ascending).

    Parameters
    ----------
    scores : Sequence[float]
        Sequence of scores.

    Returns
    -------
    List[int]
        Indices sorted by score descending, then index ascending for stability.
    """
    if scores is None:
        return []
    # Enumerate to track original indices, then sort (score desc, index asc)
    return [i for i, _ in sorted(
        enumerate(scores),
        key=lambda pair: (-pair[1], pair[0])
    )]


def rank_results(scores: Sequence[float]) -> List[int]:
    """
    Alias for rank_indices_by_scores for compatibility with tests/code that
    expect a function named `rank_results`.
    """
    return rank_indices_by_scores(scores)


def rank_descending(scores: Sequence[float]) -> List[int]:
    """
    Another alias emphasizing descending order ranking.
    """
    return rank_indices_by_scores(scores)


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------

def top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    """
    Return the indices of the top-k scores in descending order.

    Parameters
    ----------
    scores : Sequence[float]
        Sequence of scores.
    k : int
        Number of top indices to return. If k <= 0, returns [].
        If k > len(scores), returns all indices.

    Returns
    -------
    List[int]
        The indices of the top-k scores in descending order, exactly k elements
        when 0 < k <= len(scores), otherwise all indices if k exceeds the length.
    """
    if not scores or k is None or k <= 0:
        return []
    ranked = rank_indices_by_scores(scores)
    # Ensure we don't exceed the available number of scores
    k = min(k, len(ranked))
    return ranked[:k]


def top_k(scores: Sequence[float], k: int) -> List[int]:
    """
    Alias for top_k_indices for compatibility.
    """
    return top_k_indices(scores, k)


# ---------------------------------------------------------------------------
# Embedding quality metrics
# ---------------------------------------------------------------------------

def embedding_quality_metrics(embeddings: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """
    Compute simple quality metrics for a collection of embedding vectors.

    Metrics:
    - count: number of vectors.
    - dims: dimensionality (0 if not well-defined).
    - mean: mean of all elements across all vectors.
    - variance: population variance across all elements (>= 0).
    - std: standard deviation (sqrt of variance).
    - mean_norm: mean L2 norm across vectors (0 if vectors are empty).

    The function is robust to empty inputs and returns zeros for metrics
    that cannot be computed in those cases.

    Parameters
    ----------
    embeddings : Sequence[Sequence[float]]
        A sequence of embedding vectors.

    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics.
    """
    count = int(len(embeddings) if embeddings is not None else 0)
    dims = 0
    if count > 0 and embeddings[0] is not None:
        dims = int(len(embeddings[0]))

    # Flatten values
    values: List[float] = []
    norms: List[float] = []
    if embeddings:
        for vec in embeddings:
            if vec is None:
                continue
            # Collect values
            for x in vec:
                values.append(float(x))
            # Compute norm if vector is non-empty
            if len(vec) > 0:
                s = 0.0
                for x in vec:
                    fx = float(x)
                    s += fx * fx
                norms.append(math.sqrt(s))

    N = len(values)
    if N == 0:
        return {
            "count": count,
            "dims": dims,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "mean_norm": 0.0,
        }

    mean = sum(values) / float(N)
    # Population variance (non-negative)
    var_acc = 0.0
    for v in values:
        d = v - mean
        var_acc += d * d
    variance = var_acc / float(N)
    # Guard against negative due to floating point quirk
    if variance < 0.0 and abs(variance) < 1e-15:
        variance = 0.0

    std = math.sqrt(variance)
    mean_norm = sum(norms) / float(len(norms)) if norms else 0.0

    return {
        "count": count,
        "dims": dims,
        "mean": mean,
        "variance": variance,
        "std": std,
        "mean_norm": mean_norm,
    }


# Backwards-compatible alias if tests expect this exact name
embedding_quality = embedding_quality_metrics


__all__ = [
    "cosine_similarity",
    "cosine_sim",
    "chunk_text_with_overlap",
    "chunk_text",
    "chunk_document",
    "chunk_with_overlap",
    "rank_indices_by_scores",
    "rank_results",
    "rank_descending",
    "top_k_indices",
    "top_k",
    "embedding_quality_metrics",
    "embedding_quality",
]
