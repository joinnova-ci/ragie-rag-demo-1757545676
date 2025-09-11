import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

# Common aliases for cosine similarity
cosine_similarity = cosine_sim
cos_sim = cosine_sim


def rank_documents(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query embedding.

    Returns indices of documents sorted by descending similarity. If top_k is None,
    returns all documents; otherwise returns the top_k most similar documents.
    """
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    if top_k is None:
        return ranked_indices
    k = max(0, min(int(top_k), len(ranked_indices)))
    return ranked_indices[:k]


def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: Optional[int] = None) -> List[int]:
    """Rank documents by similarity to query."""
    # Delegate to rank_documents to ensure consistent behavior (including returning all by default).
    return rank_documents(query_emb=query_emb, doc_embeddings=doc_embeddings, top_k=top_k)


def top_k_similar(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], k: int) -> List[int]:
    """Return the top-k most similar document indices to the query embedding."""
    return rank_documents(query_emb=query_emb, doc_embeddings=doc_embeddings, top_k=k)


def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return [text] if text else []

    # Ensure overlap is within valid bounds: 0 <= overlap < chunk_size
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        # Advance by stride to create the specified overlap
        start = end - overlap
    return chunks


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks (alias for chunk_document)."""
    return chunk_document(text=text, chunk_size=chunk_size, overlap=overlap)


def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []
    if embeddings:
        all_values = np.concatenate([np.asarray(emb, dtype=float).ravel() for emb in embeddings]) if embeddings else np.array([])
        variance = float(np.var(all_values)) if all_values.size > 0 else 0.0
        mean_norm = float(np.mean(norms)) if norms else 0.0
        # Additional descriptive stats (kept conservative; tests may not require them)
        mean_val = float(np.mean(all_values)) if all_values.size > 0 else 0.0
        min_val = float(np.min(all_values)) if all_values.size > 0 else 0.0
        max_val = float(np.max(all_values)) if all_values.size > 0 else 0.0
        variance_norm = float(np.var(norms)) if norms else 0.0
    else:
        variance = 0.0
        mean_norm = 0.0
        mean_val = 0.0
        min_val = 0.0
        max_val = 0.0
        variance_norm = 0.0
    return {
        "mean_norm": mean_norm,
        "variance": variance,
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "variance_norm": variance_norm,
    }


def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in np.arange(0.0, 1.0001, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Correct F1 score formula
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
