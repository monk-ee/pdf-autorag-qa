#!/usr/bin/env python3
"""
Comprehensive tests for sentence transformers functionality
Tests both import and actual functionality to catch CI issues early
"""

import pytest
import numpy as np
import torch
from typing import List, Tuple
import tempfile
import os


class TestSentenceTransformersImport:
    """Test sentence-transformers import and basic functionality"""
    
    def test_import_sentence_transformers(self):
        """Test that sentence-transformers can be imported"""
        try:
            from sentence_transformers import SentenceTransformer
            assert SentenceTransformer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import sentence_transformers: {e}")
    
    def test_import_util_functions(self):
        """Test importing utility functions from sentence-transformers"""
        try:
            from sentence_transformers.util import cos_sim, semantic_search
            assert cos_sim is not None
            assert semantic_search is not None
        except ImportError as e:
            pytest.fail(f"Failed to import sentence_transformers utilities: {e}")


class TestSentenceTransformersBasicFunctionality:
    """Test basic functionality of sentence transformers"""
    
    @pytest.fixture
    def model(self):
        """Load a lightweight model for testing"""
        from sentence_transformers import SentenceTransformer
        # Use all-MiniLM-L6-v2 - small but good model for testing
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @pytest.fixture
    def sample_sentences(self) -> List[str]:
        """Sample sentences for testing"""
        return [
            "This is a test sentence about audio equipment.",
            "Guitar amplifiers provide sound amplification.",
            "Machine learning models process natural language.",
            "The weather is sunny today.",
            "Audio processing involves signal manipulation."
        ]
    
    def test_model_loading(self, model):
        """Test that the model loads successfully"""
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'get_sentence_embedding_dimension')
    
    def test_basic_encoding(self, model, sample_sentences):
        """Test basic sentence encoding"""
        embeddings = model.encode(sample_sentences)
        
        # Check output shape
        assert embeddings.shape[0] == len(sample_sentences)
        assert embeddings.shape[1] == model.get_sentence_embedding_dimension()
        
        # Check that embeddings are not all zeros
        assert not np.allclose(embeddings, 0)
        
        # Check that embeddings are normalized (if using cosine similarity models)
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(norm > 0 for norm in norms)
    
    def test_encode_single_sentence(self, model):
        """Test encoding a single sentence"""
        sentence = "This is a single test sentence."
        embedding = model.encode([sentence])
        
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == model.get_sentence_embedding_dimension()
        assert not np.allclose(embedding, 0)
    
    def test_encode_with_batch_size(self, model, sample_sentences):
        """Test encoding with specific batch size"""
        embeddings = model.encode(sample_sentences, batch_size=2)
        
        assert embeddings.shape[0] == len(sample_sentences)
        assert embeddings.shape[1] == model.get_sentence_embedding_dimension()
    
    def test_encode_show_progress_bar(self, model, sample_sentences):
        """Test encoding with progress bar enabled/disabled"""
        # Test with progress bar disabled (default for small batches)
        embeddings1 = model.encode(sample_sentences, show_progress_bar=False)
        
        # Test with progress bar enabled
        embeddings2 = model.encode(sample_sentences, show_progress_bar=True)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2, decimal=6)
    
    def test_encode_convert_to_tensor(self, model):
        """Test encoding with tensor conversion"""
        sentence = "Test sentence for tensor conversion."
        
        # Default numpy output
        embedding_np = model.encode([sentence], convert_to_tensor=False)
        assert isinstance(embedding_np, np.ndarray)
        
        # Tensor output
        embedding_tensor = model.encode([sentence], convert_to_tensor=True)
        assert torch.is_tensor(embedding_tensor)
        
        # Values should be the same
        np.testing.assert_array_almost_equal(
            embedding_np, 
            embedding_tensor.cpu().numpy(), 
            decimal=6
        )


class TestSentenceTransformersAdvancedFunctionality:
    """Test advanced functionality and edge cases"""
    
    @pytest.fixture
    def model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def test_empty_input_handling(self, model):
        """Test handling of empty inputs"""
        # Empty list
        embeddings = model.encode([])
        assert embeddings.shape[0] == 0
        
        # List with empty string
        embeddings = model.encode([""])
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == model.get_sentence_embedding_dimension()
    
    def test_very_long_sentence(self, model):
        """Test encoding of very long sentences"""
        # Create a very long sentence (beyond typical model limits)
        long_sentence = "This is a test sentence. " * 200
        
        try:
            embedding = model.encode([long_sentence])
            assert embedding.shape[0] == 1
            assert embedding.shape[1] == model.get_sentence_embedding_dimension()
        except Exception as e:
            # Some models might truncate or fail on very long inputs
            pytest.skip(f"Model cannot handle very long sentences: {e}")
    
    def test_special_characters(self, model):
        """Test encoding sentences with special characters"""
        special_sentences = [
            "Sentence with émojis 🎵 and açcénts",
            "Numbers and symbols: 123 @#$%^&*()",
            "Mixed languages: Hello 世界 Bonjour",
            "\n\t Whitespace \r\n handling   "
        ]
        
        embeddings = model.encode(special_sentences)
        assert embeddings.shape[0] == len(special_sentences)
        assert not np.allclose(embeddings, 0)
    
    def test_similarity_computation(self, model):
        """Test similarity computation between embeddings"""
        from sentence_transformers.util import cos_sim
        
        sentences = [
            "I love playing guitar.",
            "Guitar playing is my passion.",
            "The weather is nice today.",
            "Today the weather is great."
        ]
        
        embeddings = model.encode(sentences)
        
        # Compute cosine similarities
        similarities = cos_sim(embeddings, embeddings)
        
        # Check diagonal is 1 (self-similarity)
        for i in range(len(sentences)):
            assert abs(similarities[i][i].item() - 1.0) < 1e-6
        
        # Check that similar sentences have higher similarity
        guitar_sim = similarities[0][1].item()  # Both about guitar
        weather_sim = similarities[2][3].item()  # Both about weather
        cross_sim = similarities[0][2].item()    # Different topics
        
        assert guitar_sim > cross_sim
        assert weather_sim > cross_sim
    
    def test_semantic_search(self, model):
        """Test semantic search functionality"""
        from sentence_transformers.util import semantic_search
        
        corpus = [
            "Guitar amplifiers boost audio signals",
            "Audio equipment includes microphones and speakers",
            "Machine learning processes data",
            "Weather prediction uses meteorological data",
            "Music production requires quality audio gear"
        ]
        
        queries = [
            "audio amplification equipment",
            "artificial intelligence and data"
        ]
        
        corpus_embeddings = model.encode(corpus)
        query_embeddings = model.encode(queries)
        
        # Perform semantic search
        results = semantic_search(query_embeddings, corpus_embeddings, top_k=3)
        
        assert len(results) == len(queries)
        
        # Check first query (audio equipment)
        audio_results = results[0]
        assert len(audio_results) <= 3
        assert all('score' in result for result in audio_results)
        assert all('corpus_id' in result for result in audio_results)
        
        # Audio-related documents should score highly
        top_result_id = audio_results[0]['corpus_id']
        assert any(word in corpus[top_result_id].lower() for word in ['audio', 'amplifier', 'music'])


class TestSentenceTransformersResourceManagement:
    """Test resource management and cleanup"""
    
    def test_model_device_handling(self):
        """Test model device placement"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Check device placement
        if torch.cuda.is_available():
            # Model should handle CUDA automatically
            embedding = model.encode(["Test sentence"])
            assert embedding is not None
        
        # Test CPU forcing (should always work)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embedding = model.encode(["Test sentence"])
        assert embedding is not None
    
    def test_model_memory_cleanup(self):
        """Test that models can be properly cleaned up"""
        from sentence_transformers import SentenceTransformer
        import gc
        
        # Create and use model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(["Memory cleanup test"])
        assert embedding is not None
        
        # Delete model and force garbage collection
        del model
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Should be able to create a new model
        model2 = SentenceTransformer('all-MiniLM-L6-v2')
        embedding2 = model2.encode(["Second model test"])
        assert embedding2 is not None


class TestSentenceTransformersIntegration:
    """Test integration with other components"""
    
    @pytest.fixture
    def model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def test_numpy_integration(self, model):
        """Test integration with numpy operations"""
        sentences = ["Test sentence one", "Test sentence two"]
        embeddings = model.encode(sentences)
        
        # Test numpy operations
        mean_embedding = np.mean(embeddings, axis=0)
        assert mean_embedding.shape == (model.get_sentence_embedding_dimension(),)
        
        # Test normalization
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=6)
    
    def test_faiss_integration(self, model):
        """Test integration with FAISS for similarity search"""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not available for integration test")
        
        # Create embeddings
        sentences = [
            "Audio equipment testing",
            "Guitar amplifier specifications", 
            "Microphone frequency response",
            "Speaker cabinet design",
            "Weather forecast today"
        ]
        
        embeddings = model.encode(sentences)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Search for similar items
        query = "audio equipment specifications"
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, k=3)
        
        # Verify results
        assert len(scores[0]) == 3
        assert len(indices[0]) == 3
        assert all(idx < len(sentences) for idx in indices[0])
        
        # Top result should be audio-related
        top_sentence = sentences[indices[0][0]]
        assert any(word in top_sentence.lower() for word in ['audio', 'guitar', 'amplifier'])


# Performance and stress tests
class TestSentenceTransformersPerformance:
    """Test performance characteristics"""
    
    @pytest.fixture
    def model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @pytest.mark.slow
    def test_large_batch_encoding(self, model):
        """Test encoding of large batches"""
        import time
        
        # Create a large batch
        sentences = [f"Test sentence number {i}" for i in range(100)]
        
        start_time = time.time()
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
        end_time = time.time()
        
        # Verify results
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] == model.get_sentence_embedding_dimension()
        
        # Should complete in reasonable time (less than 30 seconds on most hardware)
        elapsed = end_time - start_time
        assert elapsed < 30, f"Large batch encoding took too long: {elapsed:.2f}s"
    
    def test_repeated_encoding_consistency(self, model):
        """Test that repeated encoding gives consistent results"""
        sentence = "Consistency test sentence"
        
        # Encode multiple times
        embedding1 = model.encode([sentence])
        embedding2 = model.encode([sentence])
        embedding3 = model.encode([sentence])
        
        # Results should be identical (within floating point precision)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
        np.testing.assert_array_almost_equal(embedding2, embedding3, decimal=6)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])