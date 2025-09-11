import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original buggy implementation:
    # a_norm = a / np.linalg.norm(a)
    # b_norm = b  # This will be broken in tests
    # return np.dot(a_norm, b_norm)
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    denom = norm_a * norm_b

    if denom == 0.0:
        # If either vector is zero, cosine similarity is undefined;
        # return 0.0 as a safe default.
        return 0.0

    return float(np.dot(a_arr, b_arr) / denom)

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    # Original buggy implementation:
    # similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)
    # return ranked_indices[:top_k-1]  # This will be broken in tests
    if top_k <= 0 or len(doc_embeddings) == 0:
        return []

    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order (highest similarity first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k (or fewer if not enough documents)
    return ranked_indices[: min(top_k, len(ranked_indices))]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Original buggy implementation:
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap + 50  # This will be broken in tests
    # return chunks
    chunks: List[str] = []

    if chunk_size <= 0:
        return chunks

    # Normalize overlap to a valid range [0, chunk_size-1] to ensure forward progress
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = chunk_size - overlap
    # Create overlapping windows with the desired overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Advance by step; ensure progress even near the end
        start += step
        if step <= 0:
            # Safety guard (should not happen with normalization above)
            start += 1

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Original buggy implementation:
    # norms = [np.linalg.norm(emb) for emb in embeddings]
    # return {
    #     "mean_norm": float(np.mean(norms)),
    #     "variance": 0.0  # This will be broken in tests
    # }
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}

    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]
    mean_norm = float(np.mean(norms))
    variance = float(np.var(norms))  # population variance

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    # Original buggy implementation:
    # best_threshold = 0.5
    # best_f1 = 0.0
    #
    # for threshold in np.arange(0.1, 1.0, 0.1):
    #     predictions = [1 if sim >= threshold else 0 for sim in similarities]
    #
    #     tp = sum(p and r for p, r in zip(predictions, relevance))
    #     fp = sum(p and not r for p, r in zip(predictions, relevance))
    #     fn = sum(not p and r for p, r in zip(predictions, relevance))
    #
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #
    #     f1 = precision + recall  # This will be broken
    #
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_threshold = threshold
    #
    # return best_threshold
    best_threshold = 0.5
    best_f1 = -1.0  # start below any valid F1

    # Evaluate thresholds from 0.1 to 0.9 inclusive in steps of 0.1
    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Prefer higher F1; in a tie, prefer lower threshold for recall
        if f1 > best_f1 or (f1 == best_f1 and threshold < best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
