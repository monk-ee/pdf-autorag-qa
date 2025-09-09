#!/usr/bin/env python3
"""
Test script for Adaptive RAG Pipeline
Quick verification before CI deployment
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_qa_data():
    """Create mock Q&A data for testing"""
    return [
        {
            "instruction": "What is impedance matching in audio equipment?",
            "output": "Impedance matching ensures optimal power transfer between amplifier and speaker by matching their electrical resistance values, typically 4, 8, or 16 ohms.",
            "difficulty": "intermediate"
        },
        {
            "instruction": "How does tube saturation affect guitar tone?",
            "output": "Tube saturation adds even-order harmonics and natural compression, creating warmth and musical distortion that many guitarists prefer over solid-state clipping.",
            "difficulty": "advanced"
        },
        {
            "instruction": "What is the primary function of a preamp?",
            "output": "A preamp amplifies the weak input signal from instruments or microphones to line level and provides tone shaping controls like EQ and gain.",
            "difficulty": "basic"
        },
        {
            "instruction": "How do you troubleshoot amplifier hum?",
            "output": "Check ground connections, power supply isolation, cable shielding, and proximity to interference sources like fluorescent lights or dimmer switches.",
            "difficulty": "intermediate"
        }
    ]


def test_adaptive_imports():
    """Test all imports work correctly"""
    try:
        from adaptive_rag_pipeline import (
            QueryAnalyzer, AdaptiveRetriever, 
            AdaptiveContextFormatter, AdaptiveRAGPipeline
        )
        logger.info("✅ Adaptive RAG imports: OK")
        return True
    except ImportError as e:
        logger.error(f"❌ Adaptive RAG imports: FAILED - {e}")
        return False


def test_query_analysis():
    """Test query analysis functionality"""
    try:
        from adaptive_rag_pipeline import QueryAnalyzer
        
        # Use existing config
        analyzer = QueryAnalyzer("audio_equipment_domain_questions.json")
        
        # Test in-domain query
        analysis1 = analyzer.analyze_query("What is impedance matching in guitar amplifiers?")
        assert analysis1.domain_relevance > 0.3, f"Expected medium domain relevance, got {analysis1.domain_relevance}"
        assert analysis1.query_type in ['factual', 'technical', 'conceptual'], f"Unexpected query type: {analysis1.query_type}"
        
        # Test out-domain query
        analysis2 = analyzer.analyze_query("How do you change a car tire?")
        assert analysis2.domain_relevance < 0.3, f"Expected low domain relevance, got {analysis2.domain_relevance}"
        assert analysis2.uncertainty_required, "Should require uncertainty for out-domain query"
        
        logger.info("✅ Query analysis: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Query analysis: FAILED - {e}")
        return False


def test_adaptive_retrieval():
    """Test adaptive retrieval functionality"""
    try:
        from adaptive_rag_pipeline import AdaptiveRetriever, QueryAnalyzer
        from sentence_transformers import SentenceTransformer
        
        # Mock data
        qa_data = create_mock_qa_data()
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize retriever
        retriever = AdaptiveRetriever(qa_data, embedder)
        
        # Mock analysis
        analyzer = QueryAnalyzer("audio_equipment_domain_questions.json")
        analysis = analyzer.analyze_query("What is impedance matching?")
        
        # Test retrieval
        results, strategy = retriever.retrieve_adaptive("What is impedance matching?", analysis)
        
        assert len(results) > 0, "Should retrieve at least one result"
        assert all('relevance_score' in r for r in results), "Results should have relevance scores"
        assert strategy in ['conservative', 'balanced', 'aggressive'], f"Unknown strategy: {strategy}"
        
        logger.info(f"✅ Adaptive retrieval: OK - retrieved {len(results)} items with {strategy} strategy")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive retrieval: FAILED - {e}")
        return False


def test_full_pipeline():
    """Test complete adaptive pipeline"""
    try:
        from adaptive_rag_pipeline import AdaptiveRAGPipeline
        
        # Mock data
        qa_data = create_mock_qa_data()
        
        # Initialize pipeline
        pipeline = AdaptiveRAGPipeline(qa_data, "audio_equipment_domain_questions.json")
        
        # Test in-domain query
        result1 = pipeline.process_query("What is impedance matching in guitar amplifiers?")
        assert 'analysis' in result1, "Result should include analysis"
        assert 'formatted_context' in result1, "Result should include formatted context"
        assert result1['context_quality'] > 0, "Should have positive context quality"
        
        # Test out-domain query  
        result2 = pipeline.process_query("How do you change a car tire?")
        assert result2['analysis'].uncertainty_required, "Should require uncertainty"
        assert len(result2['adaptive_adjustments']) > 0, "Should have adaptive adjustments"
        
        logger.info("✅ Full adaptive pipeline: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full adaptive pipeline: FAILED - {e}")
        return False


def main():
    """Run all adaptive RAG tests"""
    print("🧪 Testing Adaptive RAG Pipeline")
    print("=" * 50)
    
    # Check config exists
    config_file = Path("audio_equipment_domain_questions.json")
    if not config_file.exists():
        logger.error(f"❌ Config file missing: {config_file}")
        return 1
    
    tests = [
        ("Import Test", test_adaptive_imports),
        ("Query Analysis", test_query_analysis),
        ("Adaptive Retrieval", test_adaptive_retrieval),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Adaptive RAG Pipeline is working correctly.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Check errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())