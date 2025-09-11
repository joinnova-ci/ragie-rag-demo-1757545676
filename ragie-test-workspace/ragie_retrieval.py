import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure scale invariance and handle zero vectors
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0 or b_norm_val == 0:
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    if not doc_embeddings or top_k <= 0:
        return []
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results (or fewer if not enough documents)
    k = min(top_k, len(ranked_indices))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks

    # Ensure progress even if overlap is large
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [float(np.linalg.norm(emb)) for emb in embeddings]
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0

    if not similarities or not relevance or len(similarities) != len(relevance):
        return best_threshold

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
            best_threshold = float(threshold)

    return best_threshold
import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize both vectors to ensure scale invariance and handle zero vectors
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm_val = np.linalg.norm(a)
    b_norm_val = np.linalg.norm(b)
    if a_norm_val == 0 or b_norm_val == 0:
        return 0.0
    a_norm = a / a_norm_val
    b_norm = b / b_norm_val
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    if not doc_embeddings or top_k <= 0:
        return []
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Return exactly top_k results (or fewer if not enough documents)
    k = min(top_k, len(ranked_indices))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks

    # Ensure progress even if overlap is large
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }
    norms = [float(np.linalg.norm(emb)) for emb in embeddings]
    return {
        "mean_norm": float(np.mean(norms)),
        "variance": float(np.var(norms))
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0

    if not similarities or not relevance or len(similarities) != len(relevance):
        return best_threshold

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
            best_threshold = float(threshold)

    return best_threshold
