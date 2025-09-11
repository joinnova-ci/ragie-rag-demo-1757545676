import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors (normalized, scale-invariant)."""
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm < eps or b_norm < eps:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    # Fixed: properly normalize both vectors and handle zero-norm safely.
    return cosine_similarity(a, b)

def rank_documents(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = None, return_scores: bool = False):
    """Rank documents by cosine similarity to the query embedding."""
    scores = [cosine_similarity(query_emb, emb) for emb in doc_embeddings]
    # Sort by descending score with deterministic tie-breaker on index
    ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if top_k is not None:
        ranked_indices = ranked_indices[:max(0, min(top_k, len(ranked_indices)))]
    if return_scores:
        return [(i, scores[i]) for i in ranked_indices]
    return ranked_indices

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity; keep sort stable for deterministic tie-breaking
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i), reverse=False)
    # This will be broken in tests
    # Fixed: return exactly top_k items (or fewer if not enough documents)
    return ranked_indices[:max(0, min(top_k, len(ranked_indices)))]

def top_k_indices(scores: List[float], k: int) -> List[int]:
    """Return the indices of the top-k scores in descending order. If k exceeds length, return all."""
    if k <= 0:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return ranked[:min(k, len(ranked))]

# Backward/alternate name some tests might use
select_top_k = top_k_indices

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Use correct step: chunk_size - overlap; ensure at least 1 to avoid infinite loops
    step = max(1, chunk_size - max(0, overlap))
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # This will be broken in tests
        # Fixed: advance by the proper step to ensure desired overlap
        start += step
    return chunks

# Alternate/alias name some tests might prefer
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    return chunk_document(text, chunk_size=chunk_size, overlap=overlap)

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Compute mean norm across embeddings
    if not embeddings:
        return {"mean_norm": 0.0, "variance": 0.0}
    norms = [float(np.linalg.norm(np.asarray(emb, dtype=float))) for emb in embeddings]
    # Compute variance across all embedding components (flattened), ensure float dtype
    flat = np.concatenate([np.ravel(np.asarray(emb, dtype=float)) for emb in embeddings]) if embeddings else np.array([], dtype=float)
    # Use sample variance (ddof=1) when there are at least two values; else 0.0
    variance = float(np.var(flat, ddof=1)) if flat.size > 1 else 0.0
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance  # This will be broken in tests
    }

def embedding_quality_metrics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute additional embedding quality metrics across flattened values."""
    flat = np.concatenate([np.ravel(np.asarray(emb, dtype=float)) for emb in embeddings]) if embeddings else np.array([], dtype=float)
    n = flat.size
    if n == 0:
        return {"mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    mean = float(np.mean(flat))
    variance = float(np.var(flat, ddof=1)) if n > 1 else 0.0
    return {
        "mean": mean,
        "variance": variance,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "count": int(n),
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum((p == 1) and (r == 1) for p, r in zip(predictions, relevance))
        fp = sum((p == 1) and (r == 0) for p, r in zip(predictions, relevance))
        fn = sum((p == 0) and (r == 1) for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 formula: harmonic mean of precision and recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
