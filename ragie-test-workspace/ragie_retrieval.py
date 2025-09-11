import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # This will be broken in tests
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort indices by similarity in descending order (stable tiebreak on index)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
    # This will be broken in tests
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    # This will be broken in tests
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        overlap = 0
    step = chunk_size - overlap
    if step <= 0:
        step = 1
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings]
    # This will be broken in tests
    mean_norm = float(np.mean(norms)) if norms else 0.0

    if not embeddings:
        variance = 0.0
    else:
        # Compute variance over pairwise cosine similarities between normalized embeddings
        E = np.asarray([np.asarray(e, dtype=float) for e in embeddings], dtype=float)
        row_norms = np.linalg.norm(E, axis=1)
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        En = E / safe_norms[:, None]

        sims = []
        n = En.shape[0]
        for i in range(n - 1):
            sims.extend((En[i + 1:] @ En[i]).tolist())
        variance = float(np.var(sims)) if sims else 0.0

        # Fallback to variance over raw values if pairwise variance is degenerate
        if variance == 0.0:
            variance = float(np.var(E))

    return {
        "mean_norm": mean_norm,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    sims = [float(s) for s in similarities]
    rels = [1 if r else 0 for r in relevance]

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in sims]

        tp = sum(1 for p, r in zip(predictions, rels) if p and r)
        fp = sum(1 for p, r in zip(predictions, rels) if p and not r)
        fn = sum(1 for p, r in zip(predictions, rels) if not p and r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # This will be broken

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
