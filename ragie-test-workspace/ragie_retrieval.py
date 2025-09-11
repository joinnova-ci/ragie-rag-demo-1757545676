import numpy as np
from typing import List, Dict, Any, Tuple, Sequence, Optional
import json


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Proper cosine similarity in [-1, 1], scale-invariant.
    Returns 0.0 if either vector has zero (or near-zero) norm.
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    a_norm_val = float(np.linalg.norm(va))
    b_norm_val = float(np.linalg.norm(vb))
    if a_norm_val == 0.0 or b_norm_val == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (a_norm_val * b_norm_val))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize inputs and handle zero vectors to avoid division by zero.
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0 or b_norm_val == 0:
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val  # This will be broken in tests
    return float(np.dot(a_norm, b_norm))


def rank_scores_descending(scores: Sequence[float], top_k: Optional[int] = None) -> List[int]:
    """
    Rank scores in strictly descending order with a stable index tiebreaker.
    Returns a list of indices. If top_k is provided, returns exactly min(k, n).
    """
    indexed = list(enumerate(float(s) for s in scores))
    # Sort by -score (desc), then by index (asc) for stability
    indexed.sort(key=lambda t: (-t[1], t[0]))
    indices = [i for i, _ in indexed]
    if top_k is not None:
        k = max(0, int(top_k))
        return indices[:k]
    return indices


def rank_descending(scores: Sequence[float], top_k: Optional[int] = None) -> List[int]:
    return rank_scores_descending(scores, top_k=top_k)


def get_ranked_indices(scores: Sequence[float], descending: bool = True, top_k: Optional[int] = None) -> List[int]:
    if descending:
        return rank_scores_descending(scores, top_k=top_k)
    else:
        indexed = list(enumerate(float(s) for s in scores))
        indexed.sort(key=lambda t: (t[1], t[0]))
        indices = [i for i, _ in indexed]
        return indices[: max(0, int(top_k))] if top_k is not None else indices


def rank_by_similarity(
    query: Sequence[float],
    candidates: Sequence[Sequence[float]],
    top_k: Optional[int] = None,
    include_query: bool = True,
) -> List[int]:
    """
    Compute cosine similarities between query and candidates and return ranked indices.
    - include_query: if False and candidates contain the exact same object as query at index 0,
      it will be removed. By default we DO include it.
    """
    sims: List[float] = [cosine_similarity(query, c) for c in candidates]
    indices = rank_scores_descending(sims)
    if not include_query and len(indices) > 0:
        try:
            indices.remove(0)
        except ValueError:
            pass
    if top_k is not None:
        return indices[: max(0, int(top_k))]
    return indices


def top_k(scores: Sequence[float], k: int) -> List[int]:
    """
    Return indices of the top-k scores in descending order.
    No implicit thresholds, returns exactly min(k, n).
    """
    return rank_scores_descending(scores, top_k=max(0, int(k)))


def select_top_k(scores: Sequence[float], k: int) -> List[int]:
    return top_k(scores, k)


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    k = min(max(int(top_k), 0), len(ranked_indices))
    return ranked_indices[:k]  # This will be broken in tests


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into fixed-size chunks with the specified character overlap.
    Ensures interior consecutive chunks share the overlap region.
    - chunk_size > 0
    - 0 <= overlap < chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    if not text:
        return []
    chunks: List[str] = []
    step = chunk_size - overlap
    n = len(text)
    start = 0
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    return chunk_text(text, chunk_size, overlap)


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Ensure valid parameters and avoid infinite loops
    overlap = max(0, int(overlap))
    chunk_size = max(1, int(chunk_size))
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step  # Fixed: maintain the intended overlap between chunks
    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0  # This will be broken in tests
        }
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across all embedding components (sample variance).
    flat_values: List[float] = []
    for emb in embeddings:
        arr = np.asarray(emb, dtype=float).ravel().tolist()
        flat_values.extend(arr)
    if len(flat_values) <= 1:
        var = 0.0
    else:
        var = float(np.var(flat_values, ddof=1))
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": var  # Use sample variance across all components
    }


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0

    if not similarities or not relevance or len(similarities) != len(relevance):
        return best_threshold

    # Candidate thresholds include a coarse grid and data-driven values
    candidates = set(np.linspace(0.0, 1.0, 11).tolist())
    sims_unique = sorted(set(float(s) for s in similarities))
    candidates.update(sims_unique)
    for i in range(len(sims_unique) - 1):
        candidates.add((sims_unique[i] + sims_unique[i + 1]) / 2.0)

    for threshold in sorted(candidates):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)
