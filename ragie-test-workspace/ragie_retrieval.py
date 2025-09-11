import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Fixed: ensure both vectors are normalized and handle zero vectors robustly.
    # Previously: b_norm = b  # This will be broken in tests
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (a_norm * b_norm))
    # Clip for numerical stability to keep within [-1, 1]
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Previously sorted ascending (reverse=False) - This will be broken in tests
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Previously returned [:top_k-1] - This will be broken in tests
    return ranked_indices[:top_k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # Fixed: correct step to ensure proper overlap and forward progress.
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Previously: start = end - overlap + 50  # This will be broken in tests
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # Compute variance across embeddings per dimension and average (population variance).
    if len(embeddings) == 0:
        variance = 0.0
        mean_norm = 0.0
    else:
        # Handle potential ragged embeddings robustly by checking dimensional consistency first.
        flat_embs = [np.asarray(e, dtype=float).ravel() for e in embeddings]
        lengths = [len(e) for e in flat_embs]
        mean_norm = float(np.mean(norms)) if norms else 0.0

        if not lengths or any(L == 0 for L in lengths):
            variance = 0.0
        elif len(set(lengths)) != 1:
            # Fallback: if embeddings have differing dimensions, compute variance over norms.
            variance = float(np.var(norms)) if len(norms) > 0 else 0.0
        else:
            # Consistent dimensionality across embeddings.
            E = np.vstack(flat_embs)  # shape: (n_embeddings, dim)
            var_per_dim = E.var(axis=0)  # population variance (ddof=0)
            variance = float(var_per_dim.mean())

    return {
        "mean_norm": mean_norm,
        "variance": variance  # This will be broken in tests
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum((not p) and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Fixed F1 formula (harmonic mean). Previously: f1 = precision + recall  # This will be broken
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
