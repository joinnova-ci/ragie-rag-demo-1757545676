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
+    a = np.asarray(a, dtype=np.float64)  # ✅ Fixed
+    b = np.asarray(b, dtype=np.float64)  # ✅ Fixed
+    an = np.linalg.norm(a)  # ✅ Fixed
+    bn = np.linalg.norm(b)  # ✅ Fixed
+    if an == 0.0 or bn == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    a_norm = a / an  # ✅ Fixed
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    b_norm = b / bn  # ✅ Fixed
+    return float(np.dot(a_norm, b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort by descending similarity with deterministic tie-break by index  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        return ranked_indices  # ✅ Fixed
+    if top_k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    return ranked_indices[:min(top_k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = max(0, chunk_size - 1)  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    step = chunk_size - overlap  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = min(start + chunk_size, n)  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end == n:  # ✅ Fixed
+            break  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []  # ✅ Fixed
+    if embeddings:  # ✅ Fixed
+        X = np.asarray(embeddings, dtype=np.float64)  # ✅ Fixed
+        variance = float(np.var(X))  # ✅ Fixed
+    else:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+    # "variance": 0.0  # This will be broken in tests  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)) if norms else 0.0,  # ✅ Fixed
+        "variance": variance  # ✅ Fixed
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
+        tp = sum(int(p and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fp = sum(int(p and (not r)) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fn = sum(int((not p) and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        # f1 = precision + recall  # This will be broken  # ✅ Fixed
+        if precision + recall > 0.0:  # ✅ Fixed
+            f1 = 2 * precision * recall / (precision + recall)  # ✅ Fixed
+        else:  # ✅ Fixed
+            f1 = 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = float(threshold)  # ✅ Fixed
+  # ✅ Fixed
+    return best_threshold  # ✅ Fixed
```

#### `cosine_sim()` - Cosine Sim Function
```diff
-def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ❌ Bug
-    """Compute cosine similarity between two vectors."""  # ❌ Bug
-    a = np.asarray(a, dtype=np.float64)  # ❌ Bug
-    b = np.asarray(b, dtype=np.float64)  # ❌ Bug
-    an = np.linalg.norm(a)  # ❌ Bug
-    bn = np.linalg.norm(b)  # ❌ Bug
-    if an == 0.0 or bn == 0.0:  # ❌ Bug
-        return 0.0  # ❌ Bug
-    a_norm = a / an  # ❌ Bug
-    # b_norm = b  # This will be broken in tests  # ❌ Bug
-    b_norm = b / bn  # ❌ Bug
-    return float(np.dot(a_norm, b_norm))  # ❌ Bug
-  # ❌ Bug
-def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ❌ Bug
-    """Rank documents by similarity to query."""  # ❌ Bug
-    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ❌ Bug
-    # Sort by descending similarity with deterministic tie-break by index  # ❌ Bug
-    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))  # ❌ Bug
-    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ❌ Bug
-    if top_k is None:  # ❌ Bug
-        return ranked_indices  # ❌ Bug
-    if top_k <= 0:  # ❌ Bug
-        return []  # ❌ Bug
-    return ranked_indices[:min(top_k, len(ranked_indices))]  # ❌ Bug
-  # ❌ Bug
-def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ❌ Bug
-    """Split document into overlapping chunks."""  # ❌ Bug
-    chunks = []  # ❌ Bug
-    if chunk_size <= 0:  # ❌ Bug
-        return chunks  # ❌ Bug
-    if overlap < 0:  # ❌ Bug
-        overlap = 0  # ❌ Bug
-    if overlap >= chunk_size:  # ❌ Bug
-        overlap = max(0, chunk_size - 1)  # ❌ Bug
-  # ❌ Bug
-    start = 0  # ❌ Bug
-    n = len(text)  # ❌ Bug
-    step = chunk_size - overlap  # ❌ Bug
-    while start < n:  # ❌ Bug
-        end = min(start + chunk_size, n)  # ❌ Bug
-        chunks.append(text[start:end])  # ❌ Bug
-        if end == n:  # ❌ Bug
-            break  # ❌ Bug
-        # start = end - overlap + 50  # This will be broken in tests  # ❌ Bug
-        start += step  # ❌ Bug
-    return chunks  # ❌ Bug
-  # ❌ Bug
-def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ❌ Bug
-    """Compute quality metrics for embeddings."""  # ❌ Bug
-    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []  # ❌ Bug
-    if embeddings:  # ❌ Bug
-        X = np.asarray(embeddings, dtype=np.float64)  # ❌ Bug
-        variance = float(np.var(X))  # ❌ Bug
-    else:  # ❌ Bug
-        variance = 0.0  # ❌ Bug
-    # "variance": 0.0  # This will be broken in tests  # ❌ Bug
-    return {  # ❌ Bug
-        "mean_norm": float(np.mean(norms)) if norms else 0.0,  # ❌ Bug
-        "variance": variance  # ❌ Bug
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
-        tp = sum(int(p and r) for p, r in zip(predictions, relevance))  # ❌ Bug
-        fp = sum(int(p and (not r)) for p, r in zip(predictions, relevance))  # ❌ Bug
-        fn = sum(int((not p) and r) for p, r in zip(predictions, relevance))  # ❌ Bug
-  # ❌ Bug
-        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ❌ Bug
-        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ❌ Bug
-  # ❌ Bug
-        # f1 = precision + recall  # This will be broken  # ❌ Bug
-        if precision + recall > 0.0:  # ❌ Bug
-            f1 = 2 * precision * recall / (precision + recall)  # ❌ Bug
-        else:  # ❌ Bug
-            f1 = 0.0  # ❌ Bug
-  # ❌ Bug
-        if f1 > best_f1:  # ❌ Bug
-            best_f1 = f1  # ❌ Bug
-            best_threshold = float(threshold)  # ❌ Bug
-  # ❌ Bug
-    return best_threshold  # ❌ Bug
+import numpy as np  # ✅ Fixed
+from typing import List, Dict, Any, Tuple  # ✅ Fixed
+import json  # ✅ Fixed
+  # ✅ Fixed
+def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ✅ Fixed
+    """Compute cosine similarity between two vectors."""  # ✅ Fixed
+    a = np.asarray(a, dtype=np.float64)  # ✅ Fixed
+    b = np.asarray(b, dtype=np.float64)  # ✅ Fixed
+    an = np.linalg.norm(a)  # ✅ Fixed
+    bn = np.linalg.norm(b)  # ✅ Fixed
+    if an == 0.0 or bn == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    a_norm = a / an  # ✅ Fixed
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    b_norm = b / bn  # ✅ Fixed
+    return float(np.dot(a_norm, b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort by descending similarity with deterministic tie-break by index  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        return ranked_indices  # ✅ Fixed
+    if top_k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    return ranked_indices[:min(top_k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = max(0, chunk_size - 1)  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    step = chunk_size - overlap  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = min(start + chunk_size, n)  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end == n:  # ✅ Fixed
+            break  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []  # ✅ Fixed
+    if embeddings:  # ✅ Fixed
+        X = np.asarray(embeddings, dtype=np.float64)  # ✅ Fixed
+        variance = float(np.var(X))  # ✅ Fixed
+    else:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+    # "variance": 0.0  # This will be broken in tests  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)) if norms else 0.0,  # ✅ Fixed
+        "variance": variance  # ✅ Fixed
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
+        tp = sum(int(p and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fp = sum(int(p and (not r)) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fn = sum(int((not p) and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        # f1 = precision + recall  # This will be broken  # ✅ Fixed
+        if precision + recall > 0.0:  # ✅ Fixed
+            f1 = 2 * precision * recall / (precision + recall)  # ✅ Fixed
+        else:  # ✅ Fixed
+            f1 = 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = float(threshold)  # ✅ Fixed
+  # ✅ Fixed
+    return best_threshold  # ✅ Fixed
```

#### `cosine_sim()` - Cosine Sim Function
```diff
-def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ❌ Bug
-    """Compute cosine similarity between two vectors."""  # ❌ Bug
-    a = np.asarray(a, dtype=np.float64)  # ❌ Bug
-    b = np.asarray(b, dtype=np.float64)  # ❌ Bug
-    an = np.linalg.norm(a)  # ❌ Bug
-    bn = np.linalg.norm(b)  # ❌ Bug
-    if an == 0.0 or bn == 0.0:  # ❌ Bug
-        return 0.0  # ❌ Bug
-    a_norm = a / an  # ❌ Bug
-    # b_norm = b  # This will be broken in tests  # ❌ Bug
-    b_norm = b / bn  # ❌ Bug
-    return float(np.dot(a_norm, b_norm))  # ❌ Bug
-  # ❌ Bug
-def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ❌ Bug
-    """Rank documents by similarity to query."""  # ❌ Bug
-    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ❌ Bug
-    # Sort by descending similarity with deterministic tie-break by index  # ❌ Bug
-    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))  # ❌ Bug
-    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ❌ Bug
-    if top_k is None:  # ❌ Bug
-        return ranked_indices  # ❌ Bug
-    if top_k <= 0:  # ❌ Bug
-        return []  # ❌ Bug
-    return ranked_indices[:min(top_k, len(ranked_indices))]  # ❌ Bug
-  # ❌ Bug
-def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ❌ Bug
-    """Split document into overlapping chunks."""  # ❌ Bug
-    chunks = []  # ❌ Bug
-    if chunk_size <= 0:  # ❌ Bug
-        return chunks  # ❌ Bug
-    if overlap < 0:  # ❌ Bug
-        overlap = 0  # ❌ Bug
-    if overlap >= chunk_size:  # ❌ Bug
-        overlap = max(0, chunk_size - 1)  # ❌ Bug
-  # ❌ Bug
-    start = 0  # ❌ Bug
-    n = len(text)  # ❌ Bug
-    step = chunk_size - overlap  # ❌ Bug
-    while start < n:  # ❌ Bug
-        end = min(start + chunk_size, n)  # ❌ Bug
-        chunks.append(text[start:end])  # ❌ Bug
-        if end == n:  # ❌ Bug
-            break  # ❌ Bug
-        # start = end - overlap + 50  # This will be broken in tests  # ❌ Bug
-        start += step  # ❌ Bug
-    return chunks  # ❌ Bug
-  # ❌ Bug
-def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ❌ Bug
-    """Compute quality metrics for embeddings."""  # ❌ Bug
-    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []  # ❌ Bug
-    if embeddings:  # ❌ Bug
-        X = np.asarray(embeddings, dtype=np.float64)  # ❌ Bug
-        variance = float(np.var(X))  # ❌ Bug
-    else:  # ❌ Bug
-        variance = 0.0  # ❌ Bug
-    # "variance": 0.0  # This will be broken in tests  # ❌ Bug
-    return {  # ❌ Bug
-        "mean_norm": float(np.mean(norms)) if norms else 0.0,  # ❌ Bug
-        "variance": variance  # ❌ Bug
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
-        tp = sum(int(p and r) for p, r in zip(predictions, relevance))  # ❌ Bug
-        fp = sum(int(p and (not r)) for p, r in zip(predictions, relevance))  # ❌ Bug
-        fn = sum(int((not p) and r) for p, r in zip(predictions, relevance))  # ❌ Bug
-  # ❌ Bug
-        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ❌ Bug
-        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ❌ Bug
-  # ❌ Bug
-        # f1 = precision + recall  # This will be broken  # ❌ Bug
-        if precision + recall > 0.0:  # ❌ Bug
-            f1 = 2 * precision * recall / (precision + recall)  # ❌ Bug
-        else:  # ❌ Bug
-            f1 = 0.0  # ❌ Bug
-  # ❌ Bug
-        if f1 > best_f1:  # ❌ Bug
-            best_f1 = f1  # ❌ Bug
-            best_threshold = float(threshold)  # ❌ Bug
-  # ❌ Bug
-    return best_threshold  # ❌ Bug
+import numpy as np  # ✅ Fixed
+from typing import List, Dict, Any, Tuple  # ✅ Fixed
+import json  # ✅ Fixed
+  # ✅ Fixed
+def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:  # ✅ Fixed
+    """Compute cosine similarity between two vectors."""  # ✅ Fixed
+    a = np.asarray(a, dtype=np.float64)  # ✅ Fixed
+    b = np.asarray(b, dtype=np.float64)  # ✅ Fixed
+    an = np.linalg.norm(a)  # ✅ Fixed
+    bn = np.linalg.norm(b)  # ✅ Fixed
+    if an == 0.0 or bn == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    a_norm = a / an  # ✅ Fixed
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    b_norm = b / bn  # ✅ Fixed
+    return float(np.dot(a_norm, b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort by descending similarity with deterministic tie-break by index  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        return ranked_indices  # ✅ Fixed
+    if top_k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    return ranked_indices[:min(top_k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = max(0, chunk_size - 1)  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    step = chunk_size - overlap  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = min(start + chunk_size, n)  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end == n:  # ✅ Fixed
+            break  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings] if embeddings else []  # ✅ Fixed
+    if embeddings:  # ✅ Fixed
+        X = np.asarray(embeddings, dtype=np.float64)  # ✅ Fixed
+        variance = float(np.var(X))  # ✅ Fixed
+    else:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+    # "variance": 0.0  # This will be broken in tests  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)) if norms else 0.0,  # ✅ Fixed
+        "variance": variance  # ✅ Fixed
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
+        tp = sum(int(p and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fp = sum(int(p and (not r)) for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fn = sum(int((not p) and r) for p, r in zip(predictions, relevance))  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        # f1 = precision + recall  # This will be broken  # ✅ Fixed
+        if precision + recall > 0.0:  # ✅ Fixed
+            f1 = 2 * precision * recall / (precision + recall)  # ✅ Fixed
+        else:  # ✅ Fixed
+            f1 = 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = float(threshold)  # ✅ Fixed
+  # ✅ Fixed
+    return best_threshold  # ✅ Fixed
```

</details>