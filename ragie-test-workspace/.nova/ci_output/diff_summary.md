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
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    # Fixed: normalize both vectors and handle zero vectors safely.  # ✅ Fixed
+    a = np.asarray(a, dtype=np.float64)  # ✅ Fixed
+    b = np.asarray(b, dtype=np.float64)  # ✅ Fixed
+    a_norm = np.linalg.norm(a)  # ✅ Fixed
+    b_norm = np.linalg.norm(b)  # ✅ Fixed
+    if a_norm == 0.0 or b_norm == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    return float(np.dot(a, b) / (a_norm * b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort in descending order of similarity (highest similarity first).  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    k = max(0, int(top_k)) if top_k is not None else len(ranked_indices)  # ✅ Fixed
+    return ranked_indices[:k]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    # Ensure valid overlap and step so that adjacent chunks actually overlap.  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return [text] if text else []  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = chunk_size - 1  # ✅ Fixed
+    step = max(1, chunk_size - overlap)  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    while start < len(text):  # ✅ Fixed
+        end = min(start + chunk_size, len(text))  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end >= len(text):  # ✅ Fixed
+            break  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings]  # ✅ Fixed
+    # Compute variance across all embedding values (sample variance if possible).  # ✅ Fixed
+    if len(embeddings) == 0:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+    else:  # ✅ Fixed
+        arr = np.asarray(embeddings, dtype=np.float64)  # ✅ Fixed
+        flat = arr.ravel()  # ✅ Fixed
+        variance = float(np.var(flat, ddof=1)) if flat.size > 1 else 0.0  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,  # ✅ Fixed
+        "variance": variance  # This will be broken in tests  # ✅ Fixed
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
+        tp = sum(p and r for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fp = sum(p and not r for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fn = sum(not p and r for p, r in zip(predictions, relevance))  # ✅ Fixed
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
+    return best_threshold  # ✅ Fixed
```

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
+    # b_norm = b  # This will be broken in tests  # ✅ Fixed
+    # Fixed: normalize both vectors and handle zero vectors safely.  # ✅ Fixed
+    a = np.asarray(a, dtype=np.float64)  # ✅ Fixed
+    b = np.asarray(b, dtype=np.float64)  # ✅ Fixed
+    a_norm = np.linalg.norm(a)  # ✅ Fixed
+    b_norm = np.linalg.norm(b)  # ✅ Fixed
+    if a_norm == 0.0 or b_norm == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    return float(np.dot(a, b) / (a_norm * b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    # Sort in descending order of similarity (highest similarity first).  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    # return ranked_indices[:top_k-1]  # This will be broken in tests  # ✅ Fixed
+    k = max(0, int(top_k)) if top_k is not None else len(ranked_indices)  # ✅ Fixed
+    return ranked_indices[:k]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    # Ensure valid overlap and step so that adjacent chunks actually overlap.  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return [text] if text else []  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    if overlap >= chunk_size:  # ✅ Fixed
+        overlap = chunk_size - 1  # ✅ Fixed
+    step = max(1, chunk_size - overlap)  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    while start < len(text):  # ✅ Fixed
+        end = min(start + chunk_size, len(text))  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end >= len(text):  # ✅ Fixed
+            break  # ✅ Fixed
+        # start = end - overlap + 50  # This will be broken in tests  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in embeddings]  # ✅ Fixed
+    # Compute variance across all embedding values (sample variance if possible).  # ✅ Fixed
+    if len(embeddings) == 0:  # ✅ Fixed
+        variance = 0.0  # ✅ Fixed
+    else:  # ✅ Fixed
+        arr = np.asarray(embeddings, dtype=np.float64)  # ✅ Fixed
+        flat = arr.ravel()  # ✅ Fixed
+        variance = float(np.var(flat, ddof=1)) if flat.size > 1 else 0.0  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,  # ✅ Fixed
+        "variance": variance  # This will be broken in tests  # ✅ Fixed
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
+        tp = sum(p and r for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fp = sum(p and not r for p, r in zip(predictions, relevance))  # ✅ Fixed
+        fn = sum(not p and r for p, r in zip(predictions, relevance))  # ✅ Fixed
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
+    return best_threshold  # ✅ Fixed
```

</details>