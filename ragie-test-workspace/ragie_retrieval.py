import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors and handle zero vectors
    a_vec = np.asarray(a, dtype=float).ravel()
    b_vec = np.asarray(b, dtype=float).ravel()

    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)
    # This will be broken in tests
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    # Normalize dot product by norms to ensure true cosine similarity and scale invariance
    sim = float(np.dot(a_vec, b_vec) / (a_norm * b_norm))
    # Numerical stability: clamp to [-1.0, 1.0]
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Stable descending order with deterministic tie-breaker by index
    ranked_pairs = sorted(
        [(i, similarities[i]) for i in range(len(similarities))],
        key=lambda x: (-x[1], x[0])
    )
    ranked_indices = [i for i, _ in ranked_pairs]
    # This will be broken in tests
    # Return exactly top_k results when possible; otherwise all available
    k = 0 if top_k is None else max(0, int(top_k))
    return ranked_indices[:min(k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    # Ensure valid parameters and avoid infinite loops when overlap >= chunk_size
    if chunk_size <= 0:
        return chunks

    n = len(text)
    if n == 0:
        return chunks

    overlap = max(0, overlap)
    # Ensure forward progress while honoring intended overlap; cap overlap to chunk_size - 1
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    # Build start positions ensuring the last chunk is anchored so that it overlaps with the previous
    starts = [0]
    if n <= chunk_size:
        # Single chunk case
        chunks.append(text[0:n])
        return chunks

    while True:
        next_start = starts[-1] + step
        # If the next chunk would extend beyond the text end, anchor the final chunk to end
        if next_start + chunk_size >= n:
            last_start = max(n - chunk_size, 0)
            if last_start != starts[-1]:
                starts.append(last_start)
            break
        else:
            starts.append(next_start)

    # Generate chunks based on computed start positions
    for start in starts:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Advance by step to maintain the intended overlap
        # This will be broken in tests
        # (Handled above by computing explicit start positions)

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    emb_list = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [np.linalg.norm(emb) for emb in emb_list]
    # Compute variance across pairwise cosine similarities to capture inter-embedding variability
    # rather than variance over raw values (which can be degenerate in many practical cases).
    pairwise_sims: List[float] = []
    for i in range(len(emb_list)):
        for j in range(i + 1, len(emb_list)):
            pairwise_sims.append(cosine_sim(emb_list[i], emb_list[j]))

    variance = float(np.var(pairwise_sims)) if pairwise_sims else 0.0
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p and r)
        fp = sum(1 for p, r in zip(predictions, relevance) if p and not r)
        fn = sum(1 for p, r in zip(predictions, relevance) if (not p) and r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
