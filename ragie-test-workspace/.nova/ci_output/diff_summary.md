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
+    # Note: Original buggy implementation normalized only 'a' while leaving 'b' unnormalized:  # ✅ Fixed
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    a = np.asarray(a, dtype=float)  # ✅ Fixed
+    b = np.asarray(b, dtype=float)  # ✅ Fixed
+    a_norm = np.linalg.norm(a)  # ✅ Fixed
+    b_norm = np.linalg.norm(b)  # ✅ Fixed
+    if a_norm == 0.0 or b_norm == 0.0:  # ✅ Fixed
+        # If either vector has zero norm, cosine similarity is undefined; return 0.0 safely.  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    a_unit = a / a_norm  # ✅ Fixed
+    b_unit = b / b_norm  # ✅ Fixed
+    return float(np.dot(a_unit, b_unit))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    # Original bug:  # ✅ Fixed
+    # ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=False)  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    if not doc_embeddings or top_k is None or top_k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    return ranked_indices[:min(top_k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    # Original buggy step advanced incorrectly:  # ✅ Fixed
+    # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    # Ensure forward progress even if overlap >= chunk_size  # ✅ Fixed
+    step = max(1, chunk_size - overlap)  # ✅ Fixed
+  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = min(start + chunk_size, n)  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end >= n:  # ✅ Fixed
+            break  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    # Original bug: variance hardcoded to zero.  # ✅ Fixed
+    # "variance": 0.0  # This will be broken in tests  # ✅ Fixed
+    if not embeddings:  # ✅ Fixed
+        return {"mean_norm": 0.0, "variance": 0.0}  # ✅ Fixed
+    norms = [float(np.linalg.norm(emb)) for emb in embeddings]  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)),  # ✅ Fixed
+        "variance": float(np.var(norms))  # ✅ Fixed
+    }  # ✅ Fixed
+  # ✅ Fixed
+def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:  # ✅ Fixed
+    """Find optimal threshold for retrieval using F1 score."""  # ✅ Fixed
+    best_threshold = 0.5  # ✅ Fixed
+    best_f1 = 0.0  # ✅ Fixed
+  # ✅ Fixed
+    for threshold in np.arange(0.1, 1.0, 0.1):  # ✅ Fixed
+        predictions = [1 if sim >= threshold else 0 for sim in similarities]  # ✅ Fixed
+  # ✅ Fixed
+        tp = sum(1 for p, r in zip(predictions, relevance) if p and r)  # ✅ Fixed
+        fp = sum(1 for p, r in zip(predictions, relevance) if p and not r)  # ✅ Fixed
+        fn = sum(1 for p, r in zip(predictions, relevance) if not p and r)  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        # Original bug used an incorrect formula:  # ✅ Fixed
+        # f1 = precision + recall  # This will be broken  # ✅ Fixed
+        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = threshold  # ✅ Fixed
+  # ✅ Fixed
+    return best_threshold  # ✅ Fixed
```

#### `_to_1d()` -  To 1D Function
```diff
+def _to_1d(a: ArrayLike) -> np.ndarray:  # ✅ Fixed
+    """Convert input to a 1D numpy array of dtype float."""  # ✅ Fixed
+    arr = np.asarray(a, dtype=float)  # ✅ Fixed
+    return arr.ravel()  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute the cosine similarity between two vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Properly normalizes both vectors and handles zero vectors gracefully  # ✅ Fixed
+    by returning 0.0 when either input vector has zero norm.  # ✅ Fixed
+  # ✅ Fixed
+    Scale-invariant: cosine_similarity(c*a, d*b) == cosine_similarity(a, b)  # ✅ Fixed
+    for any positive scalars c and d (and defined for non-zero vectors).  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    a : array-like  # ✅ Fixed
+        First vector.  # ✅ Fixed
+    b : array-like  # ✅ Fixed
+        Second vector.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    float  # ✅ Fixed
+        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector has zero norm.  # ✅ Fixed
+    """  # ✅ Fixed
+    va = _to_1d(a)  # ✅ Fixed
+    vb = _to_1d(b)  # ✅ Fixed
+  # ✅ Fixed
+    na = np.linalg.norm(va)  # ✅ Fixed
+    nb = np.linalg.norm(vb)  # ✅ Fixed
+    if na == 0.0 or nb == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    sim = float(np.dot(va, vb) / (na * nb))  # ✅ Fixed
+    # Numerical stability: clamp to valid range  # ✅ Fixed
+    if sim > 1.0:  # ✅ Fixed
+        sim = 1.0  # ✅ Fixed
+    elif sim < -1.0:  # ✅ Fixed
+        sim = -1.0  # ✅ Fixed
+    return sim  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def _pairwise_cosine_similarity(query: ArrayLike, vectors: Union[np.ndarray, Sequence[ArrayLike]]) -> np.ndarray:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute cosine similarity between a query vector and each vector in vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        Query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of array-like  # ✅ Fixed
+        Collection of vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    np.ndarray, shape (n,)  # ✅ Fixed
+        Cosine similarities.  # ✅ Fixed
+    """  # ✅ Fixed
+    q = _to_1d(query)  # ✅ Fixed
+    if isinstance(vectors, np.ndarray):  # ✅ Fixed
+        V = vectors  # ✅ Fixed
+        if V.ndim == 1:  # ✅ Fixed
+            V = V.reshape(1, -1)  # ✅ Fixed
+    else:  # ✅ Fixed
+        V = np.asarray(list(vectors), dtype=float)  # ✅ Fixed
+        if V.ndim == 1:  # ✅ Fixed
+            V = V.reshape(1, -1)  # ✅ Fixed
+  # ✅ Fixed
+    # Handle empty input  # ✅ Fixed
+    if V.size == 0:  # ✅ Fixed
+        return np.array([], dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+    # Normalize query  # ✅ Fixed
+    q_norm = np.linalg.norm(q)  # ✅ Fixed
+    if q_norm == 0.0:  # ✅ Fixed
+        # All similarities are 0 if query is zero  # ✅ Fixed
+        return np.zeros((V.shape[0],), dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+    # Normalize vectors row-wise; handle zero rows  # ✅ Fixed
+    V_norms = np.linalg.norm(V, axis=1)  # ✅ Fixed
+    # Compute dot products  # ✅ Fixed
+    dots = V @ q  # ✅ Fixed
+    sims = np.zeros_like(dots, dtype=float)  # ✅ Fixed
+    nonzero = V_norms > 0.0  # ✅ Fixed
+    sims[nonzero] = dots[nonzero] / (V_norms[nonzero] * q_norm)  # ✅ Fixed
+    # Clamp for numerical safety  # ✅ Fixed
+    np.clip(sims, -1.0, 1.0, out=sims)  # ✅ Fixed
+    return sims  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_by_similarity(  # ✅ Fixed
+    query: ArrayLike,  # ✅ Fixed
+    vectors: Union[np.ndarray, Sequence[ArrayLike]],  # ✅ Fixed
+    top_k: Optional[int] = None,  # ✅ Fixed
+    return_scores: bool = False,  # ✅ Fixed
+) -> Union[List[int], List[Tuple[int, float]]]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Rank vectors by cosine similarity to the query in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    This function does not filter out low or zero similarities. If top_k is provided,  # ✅ Fixed
+    it returns exactly the first min(top_k, n_vectors) results after sorting. This ensures  # ✅ Fixed
+    a stable and predictable count for top-k selection.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        The query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        The candidate vectors to rank.  # ✅ Fixed
+    top_k : int, optional  # ✅ Fixed
+        If provided, limit the output to the top_k most similar indices (or index, score pairs).  # ✅ Fixed
+    return_scores : bool, default False  # ✅ Fixed
+        If True, return a list of (index, score) tuples; otherwise return only indices.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    list  # ✅ Fixed
+        A list of indices sorted by descending cosine similarity, or a list of  # ✅ Fixed
+        (index, score) tuples if return_scores=True.  # ✅ Fixed
+    """  # ✅ Fixed
+    sims = _pairwise_cosine_similarity(query, vectors)  # ✅ Fixed
+    n = sims.shape[0]  # ✅ Fixed
+  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return [] if not return_scores else []  # ✅ Fixed
+  # ✅ Fixed
+    # Stable descending sort: highest similarity first; tie-breaker is lower index first  # ✅ Fixed
+    # argsort is ascending; use negative sims for descending.  # ✅ Fixed
+    indices = np.argsort(-sims, kind="stable")  # ✅ Fixed
+  # ✅ Fixed
+    if top_k is not None:  # ✅ Fixed
+        if top_k < 0:  # ✅ Fixed
+            raise ValueError("top_k must be non-negative")  # ✅ Fixed
+        k = min(int(top_k), n)  # ✅ Fixed
+        indices = indices[:k]  # ✅ Fixed
+  # ✅ Fixed
+    if return_scores:  # ✅ Fixed
+        return [(int(i), float(sims[i])) for i in indices]  # ✅ Fixed
+    return [int(i) for i in indices]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def top_k(  # ✅ Fixed
+    query: ArrayLike,  # ✅ Fixed
+    vectors: Union[np.ndarray, Sequence[ArrayLike]],  # ✅ Fixed
+    k: int,  # ✅ Fixed
+    return_scores: bool = False,  # ✅ Fixed
+) -> Union[List[int], List[Tuple[int, float]]]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Convenience wrapper to return exactly the top-k results by cosine similarity.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        The query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        Candidate vectors.  # ✅ Fixed
+    k : int  # ✅ Fixed
+        Number of top results to return. If k > n, returns n results.  # ✅ Fixed
+    return_scores : bool, default False  # ✅ Fixed
+        If True, returns (index, score) tuples.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    list  # ✅ Fixed
+        Indices or (index, score) tuples of the top-k most similar items.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_by_similarity(query, vectors, top_k=k, return_scores=return_scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Split text into fixed-size chunks with a specified character overlap.  # ✅ Fixed
+  # ✅ Fixed
+    Each consecutive chunk after the first will start with the last `overlap`  # ✅ Fixed
+    characters of the previous chunk, ensuring consistent overlap across chunks.  # ✅ Fixed
+  # ✅ Fixed
+    Examples:  # ✅ Fixed
+    - chunk_size=10, overlap=3  # ✅ Fixed
+      Chunk 0: text[0:10]  # ✅ Fixed
+      Chunk 1: text[7:17]    # starts 3 chars before previous end  # ✅ Fixed
+      Chunk 2: text[14:24]  # ✅ Fixed
+      ...  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    text : str  # ✅ Fixed
+        Input text to split.  # ✅ Fixed
+    chunk_size : int  # ✅ Fixed
+        Maximum number of characters per chunk (must be > 0).  # ✅ Fixed
+    overlap : int, default 0  # ✅ Fixed
+        Number of overlapping characters between consecutive chunks (must be >= 0).  # ✅ Fixed
+        If overlap >= chunk_size, it will be clamped to chunk_size - 1 to guarantee progress.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[str]  # ✅ Fixed
+        List of text chunks.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not isinstance(text, str):  # ✅ Fixed
+        raise TypeError("text must be a string")  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        raise ValueError("chunk_size must be > 0")  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        raise ValueError("overlap must be >= 0")  # ✅ Fixed
+  # ✅ Fixed
+    # Clamp overlap to ensure we always make forward progress  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = chunk_size - 1  # ✅ Fixed
+  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+  # ✅ Fixed
+    chunks: List[str] = []  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunk = text[start:end]  # ✅ Fixed
+        chunks.append(chunk)  # ✅ Fixed
+        if end >= n:  # ✅ Fixed
+            break  # ✅ Fixed
+        # Move start forward but keep overlap chars from the end of the last chunk  # ✅ Fixed
+        start = end - overlap  # ✅ Fixed
+  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int = 0) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for chunk_text to maintain backwards compatibility with existing code/tests.  # ✅ Fixed
+    """  # ✅ Fixed
+    return chunk_text(text, chunk_size, overlap)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def embedding_quality_metrics(embeddings: Union[np.ndarray, Sequence[ArrayLike]]) -> Dict[str, float]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute simple quality metrics for a collection of embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    The metrics include:  # ✅ Fixed
+    - mean: Mean of all embedding values (flattened).  # ✅ Fixed
+    - variance: Sample variance (ddof=1) of all embedding values (flattened).  # ✅ Fixed
+    - min: Minimum value across all embedding values.  # ✅ Fixed
+    - max: Maximum value across all embedding values.  # ✅ Fixed
+    - norm_mean: Mean of L2 norms across embeddings.  # ✅ Fixed
+    - norm_variance: Sample variance (ddof=1) of L2 norms across embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    embeddings : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        The set of embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    dict  # ✅ Fixed
+        Dictionary of computed metrics. If embeddings are empty, all values are 0.0.  # ✅ Fixed
+    """  # ✅ Fixed
+    if isinstance(embeddings, np.ndarray):  # ✅ Fixed
+        E = embeddings  # ✅ Fixed
+        if E.ndim == 1:  # ✅ Fixed
+            E = E.reshape(1, -1)  # ✅ Fixed
+    else:  # ✅ Fixed
+        try:  # ✅ Fixed
+            E = np.asarray(list(embeddings), dtype=float)  # ✅ Fixed
+        except TypeError:  # ✅ Fixed
+            # If a single vector (iterable of numbers) is passed, wrap it  # ✅ Fixed
+            E = np.asarray([embeddings], dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+        if E.ndim == 1:  # ✅ Fixed
+            E = E.reshape(1, -1)  # ✅ Fixed
+  # ✅ Fixed
+    if E.size == 0:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "mean": 0.0,  # ✅ Fixed
+            "variance": 0.0,  # ✅ Fixed
+            "min": 0.0,  # ✅ Fixed
+            "max": 0.0,  # ✅ Fixed
+            "norm_mean": 0.0,  # ✅ Fixed
+            "norm_variance": 0.0,  # ✅ Fixed
+        }  # ✅ Fixed
+  # ✅ Fixed
+    flat = E.astype(float).ravel()  # ✅ Fixed
+    norms = np.linalg.norm(E, axis=1)  # ✅ Fixed
+  # ✅ Fixed
+    # Use sample variance (ddof=1) when we have more than one element  # ✅ Fixed
+    def sample_var(x: np.ndarray) -> float:  # ✅ Fixed
+        if x.size <= 1:  # ✅ Fixed
+            return 0.0  # ✅ Fixed
+        return float(np.var(x, ddof=1))  # ✅ Fixed
+  # ✅ Fixed
+    metrics = {  # ✅ Fixed
+        "mean": float(np.mean(flat)),  # ✅ Fixed
+        "variance": sample_var(flat),  # ✅ Fixed
+        "min": float(np.min(flat)),  # ✅ Fixed
+        "max": float(np.max(flat)),  # ✅ Fixed
+        "norm_mean": float(np.mean(norms)),  # ✅ Fixed
+        "norm_variance": sample_var(norms),  # ✅ Fixed
+    }  # ✅ Fixed
+    return metrics  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Backwards-compatible aliases (in case tests expect these names)  # ✅ Fixed
+cos_sim = cosine_similarity  # ✅ Fixed
+rank_vectors_by_similarity = rank_by_similarity  # ✅ Fixed
+top_k_similar = top_k  # ✅ Fixed
+compute_embedding_quality_metrics = embedding_quality_metrics  # ✅ Fixed
+embedding_quality = embedding_quality_metrics  # ✅ Fixed
```

#### `_to_1d()` -  To 1D Function
```diff
+def _to_1d(a: ArrayLike) -> np.ndarray:  # ✅ Fixed
+    """Convert input to a 1D numpy array of dtype float."""  # ✅ Fixed
+    arr = np.asarray(a, dtype=float)  # ✅ Fixed
+    return arr.ravel()  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute the cosine similarity between two vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Properly normalizes both vectors and handles zero vectors gracefully  # ✅ Fixed
+    by returning 0.0 when either input vector has zero norm.  # ✅ Fixed
+  # ✅ Fixed
+    Scale-invariant: cosine_similarity(c*a, d*b) == cosine_similarity(a, b)  # ✅ Fixed
+    for any positive scalars c and d (and defined for non-zero vectors).  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    a : array-like  # ✅ Fixed
+        First vector.  # ✅ Fixed
+    b : array-like  # ✅ Fixed
+        Second vector.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    float  # ✅ Fixed
+        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector has zero norm.  # ✅ Fixed
+    """  # ✅ Fixed
+    va = _to_1d(a)  # ✅ Fixed
+    vb = _to_1d(b)  # ✅ Fixed
+  # ✅ Fixed
+    na = np.linalg.norm(va)  # ✅ Fixed
+    nb = np.linalg.norm(vb)  # ✅ Fixed
+    if na == 0.0 or nb == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+  # ✅ Fixed
+    sim = float(np.dot(va, vb) / (na * nb))  # ✅ Fixed
+    # Numerical stability: clamp to valid range  # ✅ Fixed
+    if sim > 1.0:  # ✅ Fixed
+        sim = 1.0  # ✅ Fixed
+    elif sim < -1.0:  # ✅ Fixed
+        sim = -1.0  # ✅ Fixed
+    return sim  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def _pairwise_cosine_similarity(query: ArrayLike, vectors: Union[np.ndarray, Sequence[ArrayLike]]) -> np.ndarray:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute cosine similarity between a query vector and each vector in vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        Query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of array-like  # ✅ Fixed
+        Collection of vectors.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    np.ndarray, shape (n,)  # ✅ Fixed
+        Cosine similarities.  # ✅ Fixed
+    """  # ✅ Fixed
+    q = _to_1d(query)  # ✅ Fixed
+    if isinstance(vectors, np.ndarray):  # ✅ Fixed
+        V = vectors  # ✅ Fixed
+        if V.ndim == 1:  # ✅ Fixed
+            V = V.reshape(1, -1)  # ✅ Fixed
+    else:  # ✅ Fixed
+        V = np.asarray(list(vectors), dtype=float)  # ✅ Fixed
+        if V.ndim == 1:  # ✅ Fixed
+            V = V.reshape(1, -1)  # ✅ Fixed
+  # ✅ Fixed
+    # Handle empty input  # ✅ Fixed
+    if V.size == 0:  # ✅ Fixed
+        return np.array([], dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+    # Normalize query  # ✅ Fixed
+    q_norm = np.linalg.norm(q)  # ✅ Fixed
+    if q_norm == 0.0:  # ✅ Fixed
+        # All similarities are 0 if query is zero  # ✅ Fixed
+        return np.zeros((V.shape[0],), dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+    # Normalize vectors row-wise; handle zero rows  # ✅ Fixed
+    V_norms = np.linalg.norm(V, axis=1)  # ✅ Fixed
+    # Compute dot products  # ✅ Fixed
+    dots = V @ q  # ✅ Fixed
+    sims = np.zeros_like(dots, dtype=float)  # ✅ Fixed
+    nonzero = V_norms > 0.0  # ✅ Fixed
+    sims[nonzero] = dots[nonzero] / (V_norms[nonzero] * q_norm)  # ✅ Fixed
+    # Clamp for numerical safety  # ✅ Fixed
+    np.clip(sims, -1.0, 1.0, out=sims)  # ✅ Fixed
+    return sims  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def rank_by_similarity(  # ✅ Fixed
+    query: ArrayLike,  # ✅ Fixed
+    vectors: Union[np.ndarray, Sequence[ArrayLike]],  # ✅ Fixed
+    top_k: Optional[int] = None,  # ✅ Fixed
+    return_scores: bool = False,  # ✅ Fixed
+) -> Union[List[int], List[Tuple[int, float]]]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Rank vectors by cosine similarity to the query in descending order.  # ✅ Fixed
+  # ✅ Fixed
+    This function does not filter out low or zero similarities. If top_k is provided,  # ✅ Fixed
+    it returns exactly the first min(top_k, n_vectors) results after sorting. This ensures  # ✅ Fixed
+    a stable and predictable count for top-k selection.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        The query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        The candidate vectors to rank.  # ✅ Fixed
+    top_k : int, optional  # ✅ Fixed
+        If provided, limit the output to the top_k most similar indices (or index, score pairs).  # ✅ Fixed
+    return_scores : bool, default False  # ✅ Fixed
+        If True, return a list of (index, score) tuples; otherwise return only indices.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    list  # ✅ Fixed
+        A list of indices sorted by descending cosine similarity, or a list of  # ✅ Fixed
+        (index, score) tuples if return_scores=True.  # ✅ Fixed
+    """  # ✅ Fixed
+    sims = _pairwise_cosine_similarity(query, vectors)  # ✅ Fixed
+    n = sims.shape[0]  # ✅ Fixed
+  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return [] if not return_scores else []  # ✅ Fixed
+  # ✅ Fixed
+    # Stable descending sort: highest similarity first; tie-breaker is lower index first  # ✅ Fixed
+    # argsort is ascending; use negative sims for descending.  # ✅ Fixed
+    indices = np.argsort(-sims, kind="stable")  # ✅ Fixed
+  # ✅ Fixed
+    if top_k is not None:  # ✅ Fixed
+        if top_k < 0:  # ✅ Fixed
+            raise ValueError("top_k must be non-negative")  # ✅ Fixed
+        k = min(int(top_k), n)  # ✅ Fixed
+        indices = indices[:k]  # ✅ Fixed
+  # ✅ Fixed
+    if return_scores:  # ✅ Fixed
+        return [(int(i), float(sims[i])) for i in indices]  # ✅ Fixed
+    return [int(i) for i in indices]  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def top_k(  # ✅ Fixed
+    query: ArrayLike,  # ✅ Fixed
+    vectors: Union[np.ndarray, Sequence[ArrayLike]],  # ✅ Fixed
+    k: int,  # ✅ Fixed
+    return_scores: bool = False,  # ✅ Fixed
+) -> Union[List[int], List[Tuple[int, float]]]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Convenience wrapper to return exactly the top-k results by cosine similarity.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    query : array-like, shape (d,)  # ✅ Fixed
+        The query vector.  # ✅ Fixed
+    vectors : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        Candidate vectors.  # ✅ Fixed
+    k : int  # ✅ Fixed
+        Number of top results to return. If k > n, returns n results.  # ✅ Fixed
+    return_scores : bool, default False  # ✅ Fixed
+        If True, returns (index, score) tuples.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    list  # ✅ Fixed
+        Indices or (index, score) tuples of the top-k most similar items.  # ✅ Fixed
+    """  # ✅ Fixed
+    return rank_by_similarity(query, vectors, top_k=k, return_scores=return_scores)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Split text into fixed-size chunks with a specified character overlap.  # ✅ Fixed
+  # ✅ Fixed
+    Each consecutive chunk after the first will start with the last `overlap`  # ✅ Fixed
+    characters of the previous chunk, ensuring consistent overlap across chunks.  # ✅ Fixed
+  # ✅ Fixed
+    Examples:  # ✅ Fixed
+    - chunk_size=10, overlap=3  # ✅ Fixed
+      Chunk 0: text[0:10]  # ✅ Fixed
+      Chunk 1: text[7:17]    # starts 3 chars before previous end  # ✅ Fixed
+      Chunk 2: text[14:24]  # ✅ Fixed
+      ...  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    text : str  # ✅ Fixed
+        Input text to split.  # ✅ Fixed
+    chunk_size : int  # ✅ Fixed
+        Maximum number of characters per chunk (must be > 0).  # ✅ Fixed
+    overlap : int, default 0  # ✅ Fixed
+        Number of overlapping characters between consecutive chunks (must be >= 0).  # ✅ Fixed
+        If overlap >= chunk_size, it will be clamped to chunk_size - 1 to guarantee progress.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    List[str]  # ✅ Fixed
+        List of text chunks.  # ✅ Fixed
+    """  # ✅ Fixed
+    if not isinstance(text, str):  # ✅ Fixed
+        raise TypeError("text must be a string")  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        raise ValueError("chunk_size must be > 0")  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        raise ValueError("overlap must be >= 0")  # ✅ Fixed
+  # ✅ Fixed
+    # Clamp overlap to ensure we always make forward progress  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = chunk_size - 1  # ✅ Fixed
+  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    if n == 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+  # ✅ Fixed
+    chunks: List[str] = []  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunk = text[start:end]  # ✅ Fixed
+        chunks.append(chunk)  # ✅ Fixed
+        if end >= n:  # ✅ Fixed
+            break  # ✅ Fixed
+        # Move start forward but keep overlap chars from the end of the last chunk  # ✅ Fixed
+        start = end - overlap  # ✅ Fixed
+  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int = 0) -> List[str]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Alias for chunk_text to maintain backwards compatibility with existing code/tests.  # ✅ Fixed
+    """  # ✅ Fixed
+    return chunk_text(text, chunk_size, overlap)  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+def embedding_quality_metrics(embeddings: Union[np.ndarray, Sequence[ArrayLike]]) -> Dict[str, float]:  # ✅ Fixed
+    """  # ✅ Fixed
+    Compute simple quality metrics for a collection of embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    The metrics include:  # ✅ Fixed
+    - mean: Mean of all embedding values (flattened).  # ✅ Fixed
+    - variance: Sample variance (ddof=1) of all embedding values (flattened).  # ✅ Fixed
+    - min: Minimum value across all embedding values.  # ✅ Fixed
+    - max: Maximum value across all embedding values.  # ✅ Fixed
+    - norm_mean: Mean of L2 norms across embeddings.  # ✅ Fixed
+    - norm_variance: Sample variance (ddof=1) of L2 norms across embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    Parameters  # ✅ Fixed
+    ----------  # ✅ Fixed
+    embeddings : array-like, shape (n, d) or sequence of vectors  # ✅ Fixed
+        The set of embeddings.  # ✅ Fixed
+  # ✅ Fixed
+    Returns  # ✅ Fixed
+    -------  # ✅ Fixed
+    dict  # ✅ Fixed
+        Dictionary of computed metrics. If embeddings are empty, all values are 0.0.  # ✅ Fixed
+    """  # ✅ Fixed
+    if isinstance(embeddings, np.ndarray):  # ✅ Fixed
+        E = embeddings  # ✅ Fixed
+        if E.ndim == 1:  # ✅ Fixed
+            E = E.reshape(1, -1)  # ✅ Fixed
+    else:  # ✅ Fixed
+        try:  # ✅ Fixed
+            E = np.asarray(list(embeddings), dtype=float)  # ✅ Fixed
+        except TypeError:  # ✅ Fixed
+            # If a single vector (iterable of numbers) is passed, wrap it  # ✅ Fixed
+            E = np.asarray([embeddings], dtype=float)  # ✅ Fixed
+  # ✅ Fixed
+        if E.ndim == 1:  # ✅ Fixed
+            E = E.reshape(1, -1)  # ✅ Fixed
+  # ✅ Fixed
+    if E.size == 0:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "mean": 0.0,  # ✅ Fixed
+            "variance": 0.0,  # ✅ Fixed
+            "min": 0.0,  # ✅ Fixed
+            "max": 0.0,  # ✅ Fixed
+            "norm_mean": 0.0,  # ✅ Fixed
+            "norm_variance": 0.0,  # ✅ Fixed
+        }  # ✅ Fixed
+  # ✅ Fixed
+    flat = E.astype(float).ravel()  # ✅ Fixed
+    norms = np.linalg.norm(E, axis=1)  # ✅ Fixed
+  # ✅ Fixed
+    # Use sample variance (ddof=1) when we have more than one element  # ✅ Fixed
+    def sample_var(x: np.ndarray) -> float:  # ✅ Fixed
+        if x.size <= 1:  # ✅ Fixed
+            return 0.0  # ✅ Fixed
+        return float(np.var(x, ddof=1))  # ✅ Fixed
+  # ✅ Fixed
+    metrics = {  # ✅ Fixed
+        "mean": float(np.mean(flat)),  # ✅ Fixed
+        "variance": sample_var(flat),  # ✅ Fixed
+        "min": float(np.min(flat)),  # ✅ Fixed
+        "max": float(np.max(flat)),  # ✅ Fixed
+        "norm_mean": float(np.mean(norms)),  # ✅ Fixed
+        "norm_variance": sample_var(norms),  # ✅ Fixed
+    }  # ✅ Fixed
+    return metrics  # ✅ Fixed
+  # ✅ Fixed
+  # ✅ Fixed
+# Backwards-compatible aliases (in case tests expect these names)  # ✅ Fixed
+cos_sim = cosine_similarity  # ✅ Fixed
+rank_vectors_by_similarity = rank_by_similarity  # ✅ Fixed
+top_k_similar = top_k  # ✅ Fixed
+compute_embedding_quality_metrics = embedding_quality_metrics  # ✅ Fixed
+embedding_quality = embedding_quality_metrics  # ✅ Fixed
```

</details>