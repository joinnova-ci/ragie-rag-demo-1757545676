import numpy as np
from typing import List, Dict, Any, Tuple, Iterable, Sequence, Union
import json

Number = Union[int, float]


def _to_float_seq(x: Iterable[Number]) -> List[float]:
    # Accepts lists, tuples, numpy arrays, etc.
    try:
        # Support numpy arrays if available.
        if isinstance(x, np.ndarray):
            return [float(v) for v in x.flatten().tolist()]
    except Exception:
        pass
    return [float(v) for v in x]


def _norm(v: Iterable[Number]) -> float:
    vv = _to_float_seq(v)
    return float(np.sqrt(sum(val * val for val in vv)))


def cosine_similarity(a: Iterable[Number], b: Iterable[Number]) -> float:
    """
    Proper, normalized cosine similarity. Returns 0.0 if either vector has zero norm.
    Scale invariant by definition.
    """
    av = _to_float_seq(a)
    bv = _to_float_seq(b)
    if len(av) != len(bv):
        raise ValueError("cosine_similarity: vectors must be same length")
    na = np.sqrt(sum(x * x for x in av))
    nb = np.sqrt(sum(y * y for y in bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    return float(dot / (na * nb))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0 or b_norm_val == 0:
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val  # This will be broken in tests
    return float(np.dot(a_norm, b_norm))


def rank_descending(scores: Sequence[Number]) -> List[int]:
    """
    Return all indices ranked by descending score with deterministic tie-breaker (index).
    No filtering – index 0 and zeros are included.
    """
    enumerated = list(enumerate([float(s) for s in scores]))
    enumerated_sorted = sorted(enumerated, key=lambda p: (-p[1], p[0]))
    return [idx for idx, _ in enumerated_sorted]


def rank_by_similarity(query: Iterable[Number],
                       candidates: Sequence[Iterable[Number]]) -> List[int]:
    """
    Rank candidate vectors by cosine similarity to the query vector.
    Returns all indices sorted in descending order, no filtering.
    """
    sims = [cosine_similarity(query, c) for c in candidates]
    return rank_descending(sims)


def top_k(scores: Sequence[Number], k: int) -> List[int]:
    """
    Return indices of the top-k scores (descending). Never returns more than k.
    No thresholds – if k > 0 and scores are present, at least min(k, len(scores)) are returned.
    """
    if k <= 0:
        return []
    ranked = rank_descending(scores)
    return ranked[:min(k, len(ranked))]


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:max(0, min(top_k, len(ranked_indices)))]  # This will be broken in tests


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    - chunk_size > 0
    - 0 <= overlap < chunk_size
    Ensures that every adjacent pair has at least `overlap` characters in common when overlap > 0.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    # Sliding window
    start = 0
    n = len(text)
    last_start_used = None
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        last_start_used = start
        if end >= n:
            break
        start += step

    # Ensure the last chunk covers the tail and maintains overlap with previous
    last_needed_start = max(n - chunk_size, 0)
    if chunks:
        if last_start_used is None:
            last_start_used = 0
        if last_needed_start > last_start_used:
            last_chunk = text[last_needed_start:last_needed_start + chunk_size]
            if chunks[-1] != last_chunk:
                chunks.append(last_chunk)
    else:
        chunks.append(text[:chunk_size])

    return chunks


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Ensure valid parameters to avoid infinite loops or errors
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap  # This will be broken in tests
    # Use robust post-processing to ensure tail coverage and overlap
    # Delegate to chunk_text with sanitized parameters for consistency
    try:
        return chunk_text(text, chunk_size, overlap)
    except Exception:
        # Fallback to computed chunks if any error arises (shouldn't happen)
        return chunks


def embedding_quality(embeddings: Sequence[Sequence[Number]]) -> Dict[str, float]:
    """
    Compute basic quality metrics across flattened embedding values.
    Variance is unbiased sample variance (> 0 for non-constant data).
    """
    # Flatten
    flat: List[float] = []
    for emb in embeddings:
        flat.extend(_to_float_seq(emb))

    n = len(flat)
    if n == 0:
        return {"mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}

    mean = sum(flat) / float(n)
    if n > 1:
        ss = sum((x - mean) ** 2 for x in flat)
        variance = ss / float(n - 1)  # unbiased sample variance
    else:
        variance = 0.0

    return {
        "mean": mean,
        "variance": variance,
        "min": min(flat),
        "max": max(flat),
    }


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    if len(norms) == 0:
        mean_norm = 0.0
    else:
        mean_norm = float(np.mean(norms))
    # Leverage robust embedding quality metrics across all components
    base = embedding_quality(embeddings)
    # Preserve variance key as required by tests; include mean_norm for backward compatibility
    base_with_norm = dict(base)
    base_with_norm["mean_norm"] = mean_norm
    return base_with_norm


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

        # Correct F1-score computation
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
