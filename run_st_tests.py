#!/usr/bin/env python3
"""
Quick test runner for sentence-transformers functionality
Run this before CI to catch import/functionality issues early
"""

import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_sentence_transformers_import():
    """Test basic import"""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ sentence-transformers import: OK")
        return True
    except ImportError as e:
        logger.error(f"❌ sentence-transformers import: FAILED - {e}")
        traceback.print_exc()
        return False


def test_sentence_transformers_basic_functionality():
    """Test basic model loading and encoding"""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("Loading test model: all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Model loading: OK")
        
        # Test basic encoding
        sentences = ["This is a test sentence", "Another test sentence"]
        embeddings = model.encode(sentences)
        logger.info(f"✅ Basic encoding: OK - shape {embeddings.shape}")
        
        # Test similarity computation
        from sentence_transformers.util import cos_sim
        similarities = cos_sim(embeddings, embeddings)
        logger.info(f"✅ Similarity computation: OK - shape {similarities.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ sentence-transformers functionality: FAILED - {e}")
        traceback.print_exc()
        return False


def test_sentence_transformers_with_faiss():
    """Test integration with FAISS"""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create test embeddings
        sentences = ["Audio equipment", "Guitar amplifier", "Weather today"]
        embeddings = model.encode(sentences)
        
        # Build simple FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Test search
        query_embedding = model.encode(["music gear"])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, k=2)
        
        logger.info(f"✅ FAISS integration: OK - found {len(indices[0])} results")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ FAISS not available for integration test: {e}")
        return True  # Not a failure, just skip
    except Exception as e:
        logger.error(f"❌ FAISS integration: FAILED - {e}")
        traceback.print_exc()
        return False


def test_domain_evaluation_imports():
    """Test imports needed for domain evaluation"""
    try:
        # Test all imports from domain_eval_gpu.py
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from sklearn.metrics.pairwise import cosine_similarity
        from bert_score import BERTScorer
        import faiss
        import torch
        
        logger.info("✅ Domain evaluation imports: OK")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Domain evaluation imports: FAILED - {e}")
        return False


def main():
    """Run all sentence-transformers tests"""
    print("🧪 Testing sentence-transformers functionality")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_sentence_transformers_import),
        ("Basic Functionality", test_sentence_transformers_basic_functionality), 
        ("FAISS Integration", test_sentence_transformers_with_faiss),
        ("Domain Eval Imports", test_domain_evaluation_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! sentence-transformers is working correctly.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())