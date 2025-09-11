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
+    a = np.asarray(a, dtype=float).ravel()  # ✅ Fixed
+    b = np.asarray(b, dtype=float).ravel()  # ✅ Fixed
+    a_norm = np.linalg.norm(a)  # ✅ Fixed
+    b_norm = np.linalg.norm(b)  # ✅ Fixed
+    if a_norm == 0.0 or b_norm == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    return float(np.dot(a, b) / (a_norm * b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    if not doc_embeddings:  # ✅ Fixed
+        return []  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        return ranked_indices  # ✅ Fixed
+    k = int(top_k)  # ✅ Fixed
+    if k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    return ranked_indices[:min(k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if not text:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return [text]  # ✅ Fixed
+  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    # Ensure progress while maintaining requested overlap  # ✅ Fixed
+    step = (chunk_size - overlap) if overlap < chunk_size else 1  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end >= n:  # ✅ Fixed
+            break  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    if not embeddings:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "mean_norm": 0.0,  # ✅ Fixed
+            "variance": 0.0  # ✅ Fixed
+        }  # ✅ Fixed
+    arrays = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in arrays]  # ✅ Fixed
+    all_values = np.concatenate(arrays) if arrays else np.array([], dtype=float)  # ✅ Fixed
+    variance = float(np.var(all_values)) if all_values.size > 0 else 0.0  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)),  # ✅ Fixed
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
+        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)  # ✅ Fixed
+        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)  # ✅ Fixed
+        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
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
+    a = np.asarray(a, dtype=float).ravel()  # ✅ Fixed
+    b = np.asarray(b, dtype=float).ravel()  # ✅ Fixed
+    a_norm = np.linalg.norm(a)  # ✅ Fixed
+    b_norm = np.linalg.norm(b)  # ✅ Fixed
+    if a_norm == 0.0 or b_norm == 0.0:  # ✅ Fixed
+        return 0.0  # ✅ Fixed
+    return float(np.dot(a, b) / (a_norm * b_norm))  # ✅ Fixed
+  # ✅ Fixed
+def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:  # ✅ Fixed
+    """Rank documents by similarity to query."""  # ✅ Fixed
+    if not doc_embeddings:  # ✅ Fixed
+        return []  # ✅ Fixed
+    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]  # ✅ Fixed
+    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)  # ✅ Fixed
+    if top_k is None:  # ✅ Fixed
+        return ranked_indices  # ✅ Fixed
+    k = int(top_k)  # ✅ Fixed
+    if k <= 0:  # ✅ Fixed
+        return []  # ✅ Fixed
+    return ranked_indices[:min(k, len(ranked_indices))]  # ✅ Fixed
+  # ✅ Fixed
+def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:  # ✅ Fixed
+    """Split document into overlapping chunks."""  # ✅ Fixed
+    chunks = []  # ✅ Fixed
+    if not text:  # ✅ Fixed
+        return chunks  # ✅ Fixed
+    if chunk_size <= 0:  # ✅ Fixed
+        return [text]  # ✅ Fixed
+  # ✅ Fixed
+    if overlap < 0:  # ✅ Fixed
+        overlap = 0  # ✅ Fixed
+    # Ensure progress while maintaining requested overlap  # ✅ Fixed
+    step = (chunk_size - overlap) if overlap < chunk_size else 1  # ✅ Fixed
+  # ✅ Fixed
+    start = 0  # ✅ Fixed
+    n = len(text)  # ✅ Fixed
+    while start < n:  # ✅ Fixed
+        end = start + chunk_size  # ✅ Fixed
+        chunks.append(text[start:end])  # ✅ Fixed
+        if end >= n:  # ✅ Fixed
+            break  # ✅ Fixed
+        start += step  # ✅ Fixed
+    return chunks  # ✅ Fixed
+  # ✅ Fixed
+def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:  # ✅ Fixed
+    """Compute quality metrics for embeddings."""  # ✅ Fixed
+    if not embeddings:  # ✅ Fixed
+        return {  # ✅ Fixed
+            "mean_norm": 0.0,  # ✅ Fixed
+            "variance": 0.0  # ✅ Fixed
+        }  # ✅ Fixed
+    arrays = [np.asarray(emb, dtype=float).ravel() for emb in embeddings]  # ✅ Fixed
+    norms = [np.linalg.norm(emb) for emb in arrays]  # ✅ Fixed
+    all_values = np.concatenate(arrays) if arrays else np.array([], dtype=float)  # ✅ Fixed
+    variance = float(np.var(all_values)) if all_values.size > 0 else 0.0  # ✅ Fixed
+    return {  # ✅ Fixed
+        "mean_norm": float(np.mean(norms)),  # ✅ Fixed
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
+        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)  # ✅ Fixed
+        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)  # ✅ Fixed
+        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)  # ✅ Fixed
+  # ✅ Fixed
+        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # ✅ Fixed
+        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0  # ✅ Fixed
+  # ✅ Fixed
+        if f1 > best_f1:  # ✅ Fixed
+            best_f1 = f1  # ✅ Fixed
+            best_threshold = threshold  # ✅ Fixed
+  # ✅ Fixed
+    return best_threshold  # ✅ Fixed
```

</details>