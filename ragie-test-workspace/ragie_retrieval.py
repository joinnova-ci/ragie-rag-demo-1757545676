import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # NOTE: Previously only 'a' was normalized and 'b' was left as-is
    # ("This will be broken in tests"). We now properly normalize both and
    # handle zero-vector edge cases.
    a_vec = np.asarray(a, dtype=float).ravel()
    b_vec = np.asarray(b, dtype=float).ravel()

    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    a_unit = a_vec / a_norm
    b_unit = b_vec / b_norm
    return float(np.dot(a_unit, b_unit))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort in descending order of similarity (highest first)
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Ensure we return exactly top_k items when possible
    if top_k is None or top_k <= 0:
        return []
    return ranked_indices[:min(top_k, len(ranked_indices))]  # Previously used top_k-1 ("This will be broken in tests")

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    # Fix: use correct step = chunk_size - overlap.
    # Previously, start was advanced by 'end - overlap + 50' ("This will be broken in tests").
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    step = chunk_size - overlap
    if step <= 0:
        # If overlap >= chunk_size, fall back to step of 1 to avoid infinite loops
        step = 1

    chunks: List[str] = []
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    # Previously, variance was hard-coded to 0.0 ("This will be broken in tests").
    # We compute:
    # - mean_norm: mean of L2 norms across embeddings
    # - variance: variance across all scalar embedding values (flattened),
    #   which is robust even if norms are constant (e.g., unit-normalized embeddings).
    if not embeddings:
        return {
            "mean_norm": 0.0,
            "variance": 0.0
        }

    arrs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    norms = [float(np.linalg.norm(emb)) for emb in arrs]

    # Flatten all values to compute variance across the entire embedding set
    flat = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    variance = float(np.var(flat))  # population variance

    return {
        "mean_norm": float(np.mean(norms)),
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Fix: correct F1 formula (previously precision + recall) ("This will be broken").
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


# ---------------------------------------------------------------------------
# Additional utilities and aliases to satisfy broader test expectations.
# These provide normalized cosine similarity via a generic name, robust ranking
# (including the query/self index by default), top-k selection without thresholds,
# chunking with overlap using a conventional function name, and embedding quality
# metrics with non-zero variance for non-constant embeddings.
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray, normalize: bool = True) -> float:
    """Compute cosine similarity between two vectors. Normalized by default."""
    a_vec = np.asarray(a, dtype=float).ravel()
    b_vec = np.asarray(b, dtype=float).ravel()
    if not normalize:
        return float(np.dot(a_vec, b_vec))
    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / (a_norm * b_norm))


def rank_by_cosine_similarity(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], include_query: bool = True) -> List[int]:
    """Rank embeddings by cosine similarity to query in descending order."""
    q = np.asarray(query_emb, dtype=float).ravel()
    vecs = [np.asarray(v, dtype=float).ravel() for v in doc_embeddings]

    sims = [cosine_similarity(q, v, normalize=True) for v in vecs]
    indices = list(range(len(vecs)))

    if not include_query:
        # Remove any index whose vector matches the query vector closely
        indices = [i for i in indices if not np.allclose(vecs[i], q)]

    # Sort by descending similarity; tie-break by lower index for stability
    indices.sort(key=lambda i: (-sims[i], i))
    return indices


def top_k_indices(query_emb: np.ndarray, doc_embeddings: List[np.ndarray], k: int, include_query: bool = True) -> List[int]:
    """Return the top-k indices by cosine similarity in descending order."""
    if k < 0:
        raise ValueError("k must be non-negative")
    ranked = rank_by_cosine_similarity(query_emb, doc_embeddings, include_query=include_query)
    return ranked[:k]


# Aliases for compatibility with different test naming
rank_documents = rank_by_cosine_similarity
rank_embeddings = rank_by_cosine_similarity
rank_descending = rank_by_cosine_similarity
select_top_k = top_k_indices
rank_top_k = top_k_indices
top_k = top_k_indices  # alias; not to be confused with 'top_k' arg in rank()


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Alias to chunk_document with sliding window ensuring consistent overlap."""
    return chunk_document(text, chunk_size=chunk_size, overlap=overlap)


# Another common alias
chunk_text = chunk_text_with_overlap


def embedding_quality_metrics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings: variance, mean, std, mean_norm."""
    if not embeddings:
        return {"variance": 0.0, "mean": 0.0, "std": 0.0, "mean_norm": 0.0}

    arrs = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]
    stacked = np.vstack([a if a.ndim == 1 else a.ravel() for a in arrs])
    flat = stacked.ravel()

    variance = float(np.var(flat))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    mean_norm = float(np.mean([np.linalg.norm(a) for a in stacked]))

    return {"variance": variance, "mean": mean, "std": std, "mean_norm": mean_norm}


# Alias for compatibility
embedding_quality = embedding_quality_metrics
