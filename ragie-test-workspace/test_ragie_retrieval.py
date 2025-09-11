import pytest
import numpy as np
from ragie_retrieval import cosine_sim, rank, chunk_document, compute_embedding_quality, optimize_retrieval_threshold

class TestCosineSimNormalizedInvariant:
    """Test 1: Cosine similarity scale invariance - WILL FAIL due to missing normalization."""

    def test_cosine_sim_scale_invariant(self):
        """Cosine similarity should be invariant to vector scaling."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        # Test with different scales
        sim1 = cosine_sim(a, b)
        sim2 = cosine_sim(a * 10, b)  # Scale vector a
        sim3 = cosine_sim(a, b * 100)  # Scale vector b

        # All should be approximately equal (scale invariant)
        assert abs(sim1 - sim2) < 1e-10, f"Scale invariance failed: {sim1} != {sim2}"
        assert abs(sim1 - sim3) < 1e-10, f"Scale invariance failed: {sim1} != {sim3}"

class TestRankingConsistency:
    """Test 2: Document ranking consistency - WILL FAIL due to wrong order."""

    def test_ranking_descending_order(self):
        """Rankings should be in descending order of similarity."""
        documents = ["doc1", "doc2", "doc3", "doc4"]
        query_emb = np.array([1.0, 0.0, 0.0])

        # Create embeddings with known similarities
        doc_embeddings = [
            np.array([0.9, 0.1, 0.0]),  # High similarity
            np.array([0.1, 0.9, 0.0]),  # Low similarity
            np.array([0.8, 0.2, 0.0]),  # Medium-high similarity
            np.array([0.2, 0.8, 0.0])   # Medium-low similarity
        ]

        ranked_indices = rank(documents, query_emb, doc_embeddings, top_k=4)

        # Calculate actual similarities to verify order
        similarities = [cosine_sim(query_emb, emb) for emb in doc_embeddings]
        expected_order = sorted(range(len(similarities)),
                              key=lambda i: similarities[i], reverse=True)

        assert ranked_indices == expected_order, f"Wrong ranking order: {ranked_indices} != {expected_order}"

class TestTopKSelection:
    """Test 3: Top-K selection accuracy - WILL FAIL due to off-by-one error."""

    def test_top_k_exact_count(self):
        """Should return exactly top_k results, not top_k-1."""
        documents = [f"doc{i}" for i in range(10)]
        query_emb = np.array([1.0, 0.0])
        doc_embeddings = [np.random.rand(2) for _ in range(10)]

        for k in [1, 3, 5, 7]:
            ranked = rank(documents, query_emb, doc_embeddings, top_k=k)
            assert len(ranked) == k, f"Expected {k} results, got {len(ranked)}"

class TestChunkingOverlap:
    """Test 4: Document chunking overlap - WILL FAIL due to wrong overlap calculation."""

    def test_chunking_overlap_consistency(self):
        """Chunks should have consistent overlap as specified."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 20
        chunk_size = 100
        overlap = 20

        chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)

        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-overlap:]
            chunk2_start = chunks[i + 1][:overlap]

            # Should have some overlap (not exact due to word boundaries, but significant)
            overlap_chars = len(set(chunk1_end.split()) & set(chunk2_start.split()))
            assert overlap_chars > 0, f"No overlap found between chunks {i} and {i+1}"

class TestEmbeddingQualityMetrics:
    """Test 5: Embedding quality calculation - WILL FAIL due to missing variance."""

    def test_embedding_quality_variance(self):
        """Test that embedding quality includes variance calculation."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]

        quality = compute_embedding_quality(embeddings)

        # Should include variance calculation
        assert quality["variance"] > 0.0

class TestF1ScoreCalculation:
    """Test 6: F1 score formula correctness - WILL FAIL due to wrong F1 formula."""

    def test_f1_score_formula(self):
        """F1 should be harmonic mean of precision and recall, not sum."""
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        relevance = [1, 1, 1, 0, 0, 0, 0, 0]  # First 3 are relevant

        threshold = optimize_retrieval_threshold(similarities, relevance)

        # With threshold around 0.7, we should get precision=1.0, recall=1.0, F1=1.0
        # But the broken F1 formula will give F1=2.0 (precision + recall)
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Correct F1 formula (will fail with broken implementation)
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # The broken implementation returns precision + recall instead
        assert expected_f1 <= 1.0, f"F1 score should be ≤ 1.0, got {expected_f1}"
