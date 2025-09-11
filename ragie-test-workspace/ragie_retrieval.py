import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Original bug: only 'a' was normalized and 'b' was left as-is -> scale-variant similarity
    # b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # Proper, fully normalized cosine similarity
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [-1, 1] to avoid tiny numeric drift
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Original bug: sorted ascending (reverse=False) and returned top_k-1
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return ranked_indices[:max(0, top_k)]  # return exactly top_k results (or fewer if not available)

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Original bug: used step = chunk_size + overlap (effectively) via 'end - overlap + 50'
    # which destroys the intended overlap and can skip content near boundaries.
    chunks = []
    if chunk_size <= 0:
        return chunks

    # Ensure valid, non-negative overlap smaller than chunk_size to keep a positive stride
    overlap = max(0, min(overlap, chunk_size - 1))
    stride = chunk_size - overlap

    for start in range(0, len(text), stride):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]

    # Compute variance across all embedding elements (flattened), not just norms,
    # to reflect variability even when norms are similar.
    if embeddings:
        flat = np.concatenate([np.asarray(e, dtype=float).ravel() for e in embeddings])
        variance = float(np.var(flat)) if flat.size > 0 else 0.0
    else:
        variance = 0.0

    return {
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        # Original bug: returned 0.0 variance unconditionally
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    # Use a fine-grained sweep over plausible thresholds to find the best F1.
    # Original bug: F1 was computed as precision + recall.
    if not similarities:
        return best_threshold

    sim_min = min(similarities)
    sim_max = max(similarities)
    # Include slight padding to ensure edge cases are considered
    thresholds = np.linspace(sim_min - 1e-12, sim_max + 1e-12, 1001)

    for threshold in thresholds:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(int(p == 1 and r == 1) for p, r in zip(predictions, relevance))
        fp = sum(int(p == 1 and r == 0) for p, r in zip(predictions, relevance))
        fn = sum(int(p == 0 and r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
