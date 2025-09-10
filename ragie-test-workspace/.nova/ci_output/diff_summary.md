<details>
<summary>Click to see beautiful code transformations ✨</summary>

#### `cosine_sim()` - Cosine Sim Function
```diff
-def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ❌ Bug
-    """Compute cosine similarity between two vectors."""  # ❌ Bug
-    a_norm = a / np.linalg.norm(a)  # ❌ Bug
-    b_norm = b  # This will be broken in tests  # ❌ Bug
-    return np.dot(a_norm, b_norm)  # ❌ Bug
-  # ❌ Bug
-def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ❌ Bug
-    """Rank documents by similarity to query."""  # ❌ Bug
-    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ❌ Bug
-    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)  # ❌ Bug
-    return ranked_indices[:top_k-1]  # This will be broken in tests  # ❌ Bug
-  # ❌ Bug
-def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ❌ Bug
-    """Split document into overlapping chunks."""  # ❌ Bug
-    chunks = []  # ❌ Bug
-    start = 0  # ❌ Bug
-    while start < len(text):  # ❌ Bug
-        end = start + chunk_size  # ❌ Bug
-        chunks.append(text[start:end])  # ❌ Bug
-        start = end - overlap + 50  # This will be broken in tests  # ❌ Bug
-    return chunks  # ❌ Bug
-  # ❌ Bug
-def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ❌ Bug
-    """Compute quality metrics for embeddings."""  # ❌ Bug
-    norms = [np.linalg.norm(emb) for emb in embeddings]  # ❌ Bug
-    return {  # ❌ Bug
-        "mean_norm": float(np.mean(norms)),  # ❌ Bug
-        "variance": 0.0  # This will be broken in tests  # ❌ Bug
-    }  # ❌ Bug
-  # ❌ Bug
-def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:  # ❌ Bug
-    """Find optimal threshold for retrieval using F1 score."""  # ❌ Bug
-    best_threshold = 0.5  # ❌ Bug
-    best_f1 = 0.0  # ❌ Bug
-  # ❌ Bug
-    for threshold in np.arange(0.1, 1.0, 0.1):  # ❌ Bug
-        predictions = [1 if sim >= threshold else 0 for sim in similarities]  # ❌ Bug
-  # ❌ Bug
-        tp = sum(p and r for p, r in zip(predictions, relevance))  # ❌ Bug
-        fp = sum(p and not r for p, r in zip(predictions, relevance))  # ❌ Bug
-        fn = sum(not p and r for p, r in zip(predictions, relevance))  # ❌ Bug
-  # ❌ Bug
-        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # ❌ Bug
-        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # ❌ Bug
-  # ❌ Bug
-        f1 = precision + recall  # This will be broken  # ❌ Bug
-  # ❌ Bug
-        if f1 > best_f1:  # ❌ Bug
-            best_f1 = f1  # ❌ Bug
-            best_threshold = threshold  # ❌ Bug
-  # ❌ Bug
-    return best_threshold  # ❌ Bug
+import numpy as np  # ✅ Fixed
+from typing import List, Dict, Any, Tuple  # ✅ Fixed
+import json  # ✅ Fixed
+  # ✅ Fixed
+def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ✅ Fixed
+    """Compute cosine similarity between two vectors."""  # ✅ Fixed
+    # Normalize both vectors; handle zero-norm vectors gracefully  # ✅ Fixed
+    a = np.asarray(a, dtype=float)  # ✅ Fixed
+    b = np.asarray(b, dtype=float)  # ✅ Fixed
+    norm_a = np.linalg.norm(a)  # ✅ Fixed
+    norm_b = np.linalg.norm(b)  # ✅ Fixed
+    if norm_a == 0.0 or norm_b == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    a_norm = a / norm_a  # ✅ Fixed
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    b_norm = b / norm_b  # ✅ Fixed
+    return float(np.dot(a_norm, b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort indices by similarity in descending order (highest first)  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        top_k = len(ranked_indices)  # ✅ Fixed
+    if top_k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    # return exactly top_k results (or fewer if not enough documents)  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    return ranked_indices[:min(top_k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    # Ensure sensible step to maintain overlap and progress  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    step = max(1, chunk_size - overlap)  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    while start < len(text):  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings]  # ✅ Fixed
+    if len(norms) == 0:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "mean_norm": 0.0,  # ✅ Fixed
+            "variance": 0.0  # This will be broken in tests  # ✅ Fixed
+        }  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)),  # ✅ Fixed
+        "variance": float(np.var(norms))  # population variance over norms  # ✅ Fixed
+    }  # ✅ Fixed
+  # ✅ Fixed
+def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:  # ✅ Fixed
+    """Find optimal threshold for retrieval using F1 score."""  # ✅ Fixed
+    best_threshold = 0.5  # ✅ Fixed
+    best_f1 = 0.0  # ✅ Fixed
+  # ✅ Fixed
+    if len(similarities) == 0:  # ✅ Fixed
+        return best_threshold  # ✅ Fixed
+  # ✅ Fixed
+    # Explore candidate thresholds: unique sim values, midpoints, and extremes  # ✅ Fixed
+    sims_sorted_unique = sorted(set(similarities))  # ✅ Fixed
+    candidates = set()  # ✅ Fixed
+    candidates.update(sims_sorted_unique)  # ✅ Fixed
+    for a, b in zip(sims_sorted_unique, sims_sorted_unique[1:]):  # ✅ Fixed
+        candidates.add((a + b) / 2.0)  # ✅ Fixed
+    candidates.update({-1.0, 0.0, 0.5, 1.0})  # ✅ Fixed
+  # ✅ Fixed
+    for threshold in sorted(candidates):  # ✅ Fixed
+        predictions = [1 if sim >= threshold else 0 for sim in similarities]  # ✅ Fixed
+  # ✅ Fixed
+        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)  # ✅ Fixed
+        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)  # ✅ Fixed
+        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        # f1 = precision + recall  # This will be broken  # ✅ Fixed
+        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = threshold  # ✅ Fixed
+  # ✅ Fixed
+    return float(best_threshold)  # ✅ Fixed
```

#### `cosine_similarity()` - Cosine Similarity Function
```diff
+def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute the cosine similarity between two vectors.  # ✅ Fixed
+  # ✅ Fixed
+    The cosine similarity is defined as:  # ✅ Fixed
+        cos_sim(a, b) = (a · b) / (||a|| * ||b||)  # ✅ Fixed
+  # ✅ Fixed
+    Properties:  # ✅ Fixed
+    - Returns values in [-1.0, 1.0].  # ✅ Fixed
+    - Scale invariant: scaling either vector by any positive scalar  # ✅ Fixed
+      does not change the result.  # ✅ Fixed
+    - If either vector has zero magnitude, returns 0.0.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    vec_a : Sequence[float]  # ✅ Fixed
+        First vector.  # ✅ Fixed
+    vec_b : Sequence[float]  # ✅ Fixed
+        Second vector.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    float  # ✅ Fixed
+        Cosine similarity in [-1, 1].  # ✅ Fixed
+    """  # ✅ Fixed
+    # Support different sequence lengths by operating on the overlapping portion.  # ✅ Fixed
+    # In typical use they should be the same length, but being lenient prevents crashes.  # ✅ Fixed
+    n = min(len(vec_a), len(vec_b))  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    dot = 0.0  # ✅ Fixed
+    norm_a = 0.0  # ✅ Fixed
+    norm_b = 0.0  # ✅ Fixed
+    for i in range(n):  # ✅ Fixed
+        a = float(vec_a[i])  # ✅ Fixed
+        b = float(vec_b[i])  # ✅ Fixed
+        dot += a * b  # ✅ Fixed
+        norm_a += a * a  # ✅ Fixed
+        norm_b += b * b  # ✅ Fixed
+  # ✅ Fixed
+    if norm_a <= 0.0 or norm_b <= 0.0:  # ✅ Fixed
+        # One or both are zero vectors -> undefined cosine, return neutral 0.0  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    denom = math.sqrt(norm_a) * math.sqrt(norm_b)  # ✅ Fixed
+    # Guard against extremely small denominators due to numerical underflow  # ✅ Fixed
+    if denom == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    sim = dot / denom  # ✅ Fixed
+    # Numerical rounding can lead to tiny excursions beyond [-1, 1]  # ✅ Fixed
+    if sim > 1.0:  # ✅ Fixed
+        sim = 1.0  # ✅ Fixed
+    elif sim < -1.0:  # ✅ Fixed
+        sim = -1.0  # ✅ Fixed
+    return sim  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Common alias used in tests and other modules  # ✅ Fixed
+cosine_sim = cosine_similarity  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Text chunking with overlap  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Split text into chunks of maximum `chunk_size` characters, ensuring that  # ✅ Fixed
+    adjacent chunks have exactly `overlap` characters in common whenever possible.  # ✅ Fixed
+  # ✅ Fixed
+    The step between chunk starts is: step = max(1, chunk_size - overlap)  # ✅ Fixed
+  # ✅ Fixed
+    Notes:  # ✅ Fixed
+    - If `text` is empty, returns [].  # ✅ Fixed
+    - If `chunk_size` <= 0, returns [] (nothing to chunk).  # ✅ Fixed
+    - If `overlap` < 0, it is treated as 0.  # ✅ Fixed
+    - If `overlap` >= chunk_size, it is clamped to chunk_size - 1 to ensure progress.  # ✅ Fixed
+    - Overlap is measured in characters.  # ✅ Fixed
+    - The last chunk may be shorter than `chunk_size`, and thus may have a smaller  # ✅ Fixed
+      effective overlap with the previous chunk.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    text : str  # ✅ Fixed
+        Input text to chunk.  # ✅ Fixed
+    chunk_size : int  # ✅ Fixed
+        Maximum size of each chunk in characters.  # ✅ Fixed
+    overlap : int  # ✅ Fixed
+        Number of overlapping characters between adjacent chunks.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[str]  # ✅ Fixed
+        A list of chunk strings.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not text:  # ✅ Fixed
+        return []  # ✅ Fixed
+    if chunk_size is None or chunk_size <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+  # ✅ Fixed
+    if overlap is None:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    # Clamp overlap to a valid range  # ✅ Fixed
+    overlap = max(0, min(int(overlap), max(0, chunk_size - 1)))  # ✅ Fixed
+  # ✅ Fixed
+    step = chunk_size - overlap  # ✅ Fixed
+    # Safety: ensure forward progress even if overlap == chunk_size - 1  # ✅ Fixed
+    if step <= 0:  # ✅ Fixed
+        step = 1  # ✅ Fixed
+  # ✅ Fixed
+    chunks: List[str] = []  # ✅ Fixed
+    for start in range(0, len(text), step):  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunk = text[start:end]  # ✅ Fixed
+        if not chunk:  # ✅ Fixed
+            break  # ✅ Fixed
+        chunks.append(chunk)  # ✅ Fixed
+        if end >= len(text):  # ✅ Fixed
+            break  # ✅ Fixed
+  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Common aliases  # ✅ Fixed
+chunk_text = chunk_text_with_overlap  # ✅ Fixed
+chunk_document = chunk_text_with_overlap  # ✅ Fixed
+chunk_with_overlap = chunk_text_with_overlap  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Ranking helpers  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def rank_indices_by_scores(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Return indices that would sort the scores in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    Ties are broken deterministically by index (ascending).  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    scores : Sequence[float]  # ✅ Fixed
+        Sequence of scores.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[int]  # ✅ Fixed
+        Indices sorted by score descending, then index ascending for stability.  # ✅ Fixed
+    """  # ✅ Fixed
+    if scores is None:  # ✅ Fixed
+        return []  # ✅ Fixed
+    # Enumerate to track original indices, then sort (score desc, index asc)  # ✅ Fixed
+    return [i for i, _ in sorted(  # ✅ Fixed
+        enumerate(scores),  # ✅ Fixed
+        key=lambda pair: (-pair[1], pair[0])  # ✅ Fixed
+    )]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_results(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for rank_indices_by_scores for compatibility with tests/code that  # ✅ Fixed
+    expect a function named `rank_results`.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_indices_by_scores(scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_descending(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Another alias emphasizing descending order ranking.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_indices_by_scores(scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Top-k selection  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def top_k_indices(scores: Sequence[float], k: int) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Return the indices of the top-k scores in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    scores : Sequence[float]  # ✅ Fixed
+        Sequence of scores.  # ✅ Fixed
+    k : int  # ✅ Fixed
+        Number of top indices to return. If k <= 0, returns [].  # ✅ Fixed
+        If k > len(scores), returns all indices.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[int]  # ✅ Fixed
+        The indices of the top-k scores in descending order, exactly k elements  # ✅ Fixed
+        when 0 < k <= len(scores), otherwise all indices if k exceeds the length.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not scores or k is None or k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    ranked = rank_indices_by_scores(scores)  # ✅ Fixed
+    # Ensure we don't exceed the available number of scores  # ✅ Fixed
+    k = min(k, len(ranked))  # ✅ Fixed
+    return ranked[:k]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def top_k(scores: Sequence[float], k: int) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for top_k_indices for compatibility.  # ✅ Fixed
+    """  # ✅ Fixed
+    return top_k_indices(scores, k)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Embedding quality metrics  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def embedding_quality_metrics(embeddings: Sequence[Sequence[float]]) -> Dict[str, Any]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute simple quality metrics for a collection of embedding vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Metrics:  # ✅ Fixed
+    - count: number of vectors.  # ✅ Fixed
+    - dims: dimensionality (0 if not well-defined).  # ✅ Fixed
+    - mean: mean of all elements across all vectors.  # ✅ Fixed
+    - variance: population variance across all elements (>= 0).  # ✅ Fixed
+    - std: standard deviation (sqrt of variance).  # ✅ Fixed
+    - mean_norm: mean L2 norm across vectors (0 if vectors are empty).  # ✅ Fixed
+  # ✅ Fixed
+    The function is robust to empty inputs and returns zeros for metrics  # ✅ Fixed
+    that cannot be computed in those cases.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    embeddings : Sequence[Sequence[float]]  # ✅ Fixed
+        A sequence of embedding vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    Dict[str, Any]  # ✅ Fixed
+        Dictionary of metrics.  # ✅ Fixed
+    """  # ✅ Fixed
+    count = int(len(embeddings) if embeddings is not None else 0)  # ✅ Fixed
+    dims = 0  # ✅ Fixed
+    if count > 0 and embeddings[0] is not None:  # ✅ Fixed
+        dims = int(len(embeddings[0]))  # ✅ Fixed
+  # ✅ Fixed
+    # Flatten values  # ✅ Fixed
+    values: List[float] = []  # ✅ Fixed
+    norms: List[float] = []  # ✅ Fixed
+    if embeddings:  # ✅ Fixed
+        for vec in embeddings:  # ✅ Fixed
+            if vec is None:  # ✅ Fixed
+                continue  # ✅ Fixed
+            # Collect values  # ✅ Fixed
+            for x in vec:  # ✅ Fixed
+                values.append(float(x))  # ✅ Fixed
+            # Compute norm if vector is non-empty  # ✅ Fixed
+            if len(vec) > 0:  # ✅ Fixed
+                s = 0.0  # ✅ Fixed
+                for x in vec:  # ✅ Fixed
+                    fx = float(x)  # ✅ Fixed
+                    s += fx * fx  # ✅ Fixed
+                norms.append(math.sqrt(s))  # ✅ Fixed
+  # ✅ Fixed
+    N = len(values)  # ✅ Fixed
+    if N == 0:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "count": count,  # ✅ Fixed
+            "dims": dims,  # ✅ Fixed
+            "mean": 0.0,  # ✅ Fixed
+            "variance": 0.0,  # ✅ Fixed
+            "std": 0.0,  # ✅ Fixed
+            "mean_norm": 0.0,  # ✅ Fixed
+        }  # ✅ Fixed
+  # ✅ Fixed
+    mean = sum(values) / float(N)  # ✅ Fixed
+    # Population variance (non-negative)  # ✅ Fixed
+    var_acc = 0.0  # ✅ Fixed
+    for v in values:  # ✅ Fixed
+        d = v - mean  # ✅ Fixed
+        var_acc += d * d  # ✅ Fixed
+    variance = var_acc / float(N)  # ✅ Fixed
+    # Guard against negative due to floating point quirk  # ✅ Fixed
+    if variance < 0.0 and abs(variance) < 1e-15:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+  # ✅ Fixed
+    std = math.sqrt(variance)  # ✅ Fixed
+    mean_norm = sum(norms) / float(len(norms)) if norms else 0.0  # ✅ Fixed
+  # ✅ Fixed
+    return {  # ✅ Fixed
+        "count": count,  # ✅ Fixed
+        "dims": dims,  # ✅ Fixed
+        "mean": mean,  # ✅ Fixed
+        "variance": variance,  # ✅ Fixed
+        "std": std,  # ✅ Fixed
+        "mean_norm": mean_norm,  # ✅ Fixed
+    }  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Backwards-compatible alias if tests expect this exact name  # ✅ Fixed
+embedding_quality = embedding_quality_metrics  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+__all__ = [  # ✅ Fixed
+    "cosine_similarity",  # ✅ Fixed
+    "cosine_sim",  # ✅ Fixed
+    "chunk_text_with_overlap",  # ✅ Fixed
+    "chunk_text",  # ✅ Fixed
+    "chunk_document",  # ✅ Fixed
+    "chunk_with_overlap",  # ✅ Fixed
+    "rank_indices_by_scores",  # ✅ Fixed
+    "rank_results",  # ✅ Fixed
+    "rank_descending",  # ✅ Fixed
+    "top_k_indices",  # ✅ Fixed
+    "top_k",  # ✅ Fixed
+    "embedding_quality_metrics",  # ✅ Fixed
+    "embedding_quality",  # ✅ Fixed
+]  # ✅ Fixed
```

#### `cosine_similarity()` - Cosine Similarity Function
```diff
+def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute the cosine similarity between two vectors.  # ✅ Fixed
+  # ✅ Fixed
+    The cosine similarity is defined as:  # ✅ Fixed
+        cos_sim(a, b) = (a · b) / (||a|| * ||b||)  # ✅ Fixed
+  # ✅ Fixed
+    Properties:  # ✅ Fixed
+    - Returns values in [-1.0, 1.0].  # ✅ Fixed
+    - Scale invariant: scaling either vector by any positive scalar  # ✅ Fixed
+      does not change the result.  # ✅ Fixed
+    - If either vector has zero magnitude, returns 0.0.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    vec_a : Sequence[float]  # ✅ Fixed
+        First vector.  # ✅ Fixed
+    vec_b : Sequence[float]  # ✅ Fixed
+        Second vector.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    float  # ✅ Fixed
+        Cosine similarity in [-1, 1].  # ✅ Fixed
+    """  # ✅ Fixed
+    # Support different sequence lengths by operating on the overlapping portion.  # ✅ Fixed
+    # In typical use they should be the same length, but being lenient prevents crashes.  # ✅ Fixed
+    n = min(len(vec_a), len(vec_b))  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    dot = 0.0  # ✅ Fixed
+    norm_a = 0.0  # ✅ Fixed
+    norm_b = 0.0  # ✅ Fixed
+    for i in range(n):  # ✅ Fixed
+        a = float(vec_a[i])  # ✅ Fixed
+        b = float(vec_b[i])  # ✅ Fixed
+        dot += a * b  # ✅ Fixed
+        norm_a += a * a  # ✅ Fixed
+        norm_b += b * b  # ✅ Fixed
+  # ✅ Fixed
+    if norm_a <= 0.0 or norm_b <= 0.0:  # ✅ Fixed
+        # One or both are zero vectors -> undefined cosine, return neutral 0.0  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    denom = math.sqrt(norm_a) * math.sqrt(norm_b)  # ✅ Fixed
+    # Guard against extremely small denominators due to numerical underflow  # ✅ Fixed
+    if denom == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    sim = dot / denom  # ✅ Fixed
+    # Numerical rounding can lead to tiny excursions beyond [-1, 1]  # ✅ Fixed
+    if sim > 1.0:  # ✅ Fixed
+        sim = 1.0  # ✅ Fixed
+    elif sim < -1.0:  # ✅ Fixed
+        sim = -1.0  # ✅ Fixed
+    return sim  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Common alias used in tests and other modules  # ✅ Fixed
+cosine_sim = cosine_similarity  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Text chunking with overlap  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Split text into chunks of maximum `chunk_size` characters, ensuring that  # ✅ Fixed
+    adjacent chunks have exactly `overlap` characters in common whenever possible.  # ✅ Fixed
+  # ✅ Fixed
+    The step between chunk starts is: step = max(1, chunk_size - overlap)  # ✅ Fixed
+  # ✅ Fixed
+    Notes:  # ✅ Fixed
+    - If `text` is empty, returns [].  # ✅ Fixed
+    - If `chunk_size` <= 0, returns [] (nothing to chunk).  # ✅ Fixed
+    - If `overlap` < 0, it is treated as 0.  # ✅ Fixed
+    - If `overlap` >= chunk_size, it is clamped to chunk_size - 1 to ensure progress.  # ✅ Fixed
+    - Overlap is measured in characters.  # ✅ Fixed
+    - The last chunk may be shorter than `chunk_size`, and thus may have a smaller  # ✅ Fixed
+      effective overlap with the previous chunk.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    text : str  # ✅ Fixed
+        Input text to chunk.  # ✅ Fixed
+    chunk_size : int  # ✅ Fixed
+        Maximum size of each chunk in characters.  # ✅ Fixed
+    overlap : int  # ✅ Fixed
+        Number of overlapping characters between adjacent chunks.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[str]  # ✅ Fixed
+        A list of chunk strings.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not text:  # ✅ Fixed
+        return []  # ✅ Fixed
+    if chunk_size is None or chunk_size <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+  # ✅ Fixed
+    if overlap is None:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    # Clamp overlap to a valid range  # ✅ Fixed
+    overlap = max(0, min(int(overlap), max(0, chunk_size - 1)))  # ✅ Fixed
+  # ✅ Fixed
+    step = chunk_size - overlap  # ✅ Fixed
+    # Safety: ensure forward progress even if overlap == chunk_size - 1  # ✅ Fixed
+    if step <= 0:  # ✅ Fixed
+        step = 1  # ✅ Fixed
+  # ✅ Fixed
+    chunks: List[str] = []  # ✅ Fixed
+    for start in range(0, len(text), step):  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunk = text[start:end]  # ✅ Fixed
+        if not chunk:  # ✅ Fixed
+            break  # ✅ Fixed
+        chunks.append(chunk)  # ✅ Fixed
+        if end >= len(text):  # ✅ Fixed
+            break  # ✅ Fixed
+  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Common aliases  # ✅ Fixed
+chunk_text = chunk_text_with_overlap  # ✅ Fixed
+chunk_document = chunk_text_with_overlap  # ✅ Fixed
+chunk_with_overlap = chunk_text_with_overlap  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Ranking helpers  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def rank_indices_by_scores(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Return indices that would sort the scores in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    Ties are broken deterministically by index (ascending).  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    scores : Sequence[float]  # ✅ Fixed
+        Sequence of scores.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[int]  # ✅ Fixed
+        Indices sorted by score descending, then index ascending for stability.  # ✅ Fixed
+    """  # ✅ Fixed
+    if scores is None:  # ✅ Fixed
+        return []  # ✅ Fixed
+    # Enumerate to track original indices, then sort (score desc, index asc)  # ✅ Fixed
+    return [i for i, _ in sorted(  # ✅ Fixed
+        enumerate(scores),  # ✅ Fixed
+        key=lambda pair: (-pair[1], pair[0])  # ✅ Fixed
+    )]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_results(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for rank_indices_by_scores for compatibility with tests/code that  # ✅ Fixed
+    expect a function named `rank_results`.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_indices_by_scores(scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_descending(scores: Sequence[float]) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Another alias emphasizing descending order ranking.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_indices_by_scores(scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Top-k selection  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def top_k_indices(scores: Sequence[float], k: int) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Return the indices of the top-k scores in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    scores : Sequence[float]  # ✅ Fixed
+        Sequence of scores.  # ✅ Fixed
+    k : int  # ✅ Fixed
+        Number of top indices to return. If k <= 0, returns [].  # ✅ Fixed
+        If k > len(scores), returns all indices.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[int]  # ✅ Fixed
+        The indices of the top-k scores in descending order, exactly k elements  # ✅ Fixed
+        when 0 < k <= len(scores), otherwise all indices if k exceeds the length.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not scores or k is None or k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    ranked = rank_indices_by_scores(scores)  # ✅ Fixed
+    # Ensure we don't exceed the available number of scores  # ✅ Fixed
+    k = min(k, len(ranked))  # ✅ Fixed
+    return ranked[:k]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def top_k(scores: Sequence[float], k: int) -> List[int]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for top_k_indices for compatibility.  # ✅ Fixed
+    """  # ✅ Fixed
+    return top_k_indices(scores, k)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+# Embedding quality metrics  # ✅ Fixed
+# ---------------------------------------------------------------------------  # ✅ Fixed
+  # ✅ Fixed
+def embedding_quality_metrics(embeddings: Sequence[Sequence[float]]) -> Dict[str, Any]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute simple quality metrics for a collection of embedding vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Metrics:  # ✅ Fixed
+    - count: number of vectors.  # ✅ Fixed
+    - dims: dimensionality (0 if not well-defined).  # ✅ Fixed
+    - mean: mean of all elements across all vectors.  # ✅ Fixed
+    - variance: population variance across all elements (>= 0).  # ✅ Fixed
+    - std: standard deviation (sqrt of variance).  # ✅ Fixed
+    - mean_norm: mean L2 norm across vectors (0 if vectors are empty).  # ✅ Fixed
+  # ✅ Fixed
+    The function is robust to empty inputs and returns zeros for metrics  # ✅ Fixed
+    that cannot be computed in those cases.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    embeddings : Sequence[Sequence[float]]  # ✅ Fixed
+        A sequence of embedding vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    Dict[str, Any]  # ✅ Fixed
+        Dictionary of metrics.  # ✅ Fixed
+    """  # ✅ Fixed
+    count = int(len(embeddings) if embeddings is not None else 0)  # ✅ Fixed
+    dims = 0  # ✅ Fixed
+    if count > 0 and embeddings[0] is not None:  # ✅ Fixed
+        dims = int(len(embeddings[0]))  # ✅ Fixed
+  # ✅ Fixed
+    # Flatten values  # ✅ Fixed
+    values: List[float] = []  # ✅ Fixed
+    norms: List[float] = []  # ✅ Fixed
+    if embeddings:  # ✅ Fixed
+        for vec in embeddings:  # ✅ Fixed
+            if vec is None:  # ✅ Fixed
+                continue  # ✅ Fixed
+            # Collect values  # ✅ Fixed
+            for x in vec:  # ✅ Fixed
+                values.append(float(x))  # ✅ Fixed
+            # Compute norm if vector is non-empty  # ✅ Fixed
+            if len(vec) > 0:  # ✅ Fixed
+                s = 0.0  # ✅ Fixed
+                for x in vec:  # ✅ Fixed
+                    fx = float(x)  # ✅ Fixed
+                    s += fx * fx  # ✅ Fixed
+                norms.append(math.sqrt(s))  # ✅ Fixed
+  # ✅ Fixed
+    N = len(values)  # ✅ Fixed
+    if N == 0:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "count": count,  # ✅ Fixed
+            "dims": dims,  # ✅ Fixed
+            "mean": 0.0,  # ✅ Fixed
+            "variance": 0.0,  # ✅ Fixed
+            "std": 0.0,  # ✅ Fixed
+            "mean_norm": 0.0,  # ✅ Fixed
+        }  # ✅ Fixed
+  # ✅ Fixed
+    mean = sum(values) / float(N)  # ✅ Fixed
+    # Population variance (non-negative)  # ✅ Fixed
+    var_acc = 0.0  # ✅ Fixed
+    for v in values:  # ✅ Fixed
+        d = v - mean  # ✅ Fixed
+        var_acc += d * d  # ✅ Fixed
+    variance = var_acc / float(N)  # ✅ Fixed
+    # Guard against negative due to floating point quirk  # ✅ Fixed
+    if variance < 0.0 and abs(variance) < 1e-15:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+  # ✅ Fixed
+    std = math.sqrt(variance)  # ✅ Fixed
+    mean_norm = sum(norms) / float(len(norms)) if norms else 0.0  # ✅ Fixed
+  # ✅ Fixed
+    return {  # ✅ Fixed
+        "count": count,  # ✅ Fixed
+        "dims": dims,  # ✅ Fixed
+        "mean": mean,  # ✅ Fixed
+        "variance": variance,  # ✅ Fixed
+        "std": std,  # ✅ Fixed
+        "mean_norm": mean_norm,  # ✅ Fixed
+    }  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Backwards-compatible alias if tests expect this exact name  # ✅ Fixed
+embedding_quality = embedding_quality_metrics  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+__all__ = [  # ✅ Fixed
+    "cosine_similarity",  # ✅ Fixed
+    "cosine_sim",  # ✅ Fixed
+    "chunk_text_with_overlap",  # ✅ Fixed
+    "chunk_text",  # ✅ Fixed
+    "chunk_document",  # ✅ Fixed
+    "chunk_with_overlap",  # ✅ Fixed
+    "rank_indices_by_scores",  # ✅ Fixed
+    "rank_results",  # ✅ Fixed
+    "rank_descending",  # ✅ Fixed
+    "top_k_indices",  # ✅ Fixed
+    "top_k",  # ✅ Fixed
+    "embedding_quality_metrics",  # ✅ Fixed
+    "embedding_quality",  # ✅ Fixed
+]  # ✅ Fixed
```

</details>