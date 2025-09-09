#!/usr/bin/env python3
"""
Quick test to validate Enhanced Adaptive RAG fixes
"""

import json
from pathlib import Path
import logging
from adaptive_rag_pipeline import AdaptiveRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_pipeline_fixes():
    """Test the enhanced pipeline with fixes"""
    
    # Load a small sample of Q&A pairs for testing
    qa_pairs_file = Path("rag_input/selected_qa_pairs.json")
    
    if not qa_pairs_file.exists():
        print(f"❌ Test file not found: {qa_pairs_file}")
        return False
    
    print("🧪 Testing Enhanced Adaptive RAG Pipeline fixes...")
    
    try:
        # Load Q&A pairs
        with open(qa_pairs_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        print(f"📊 Loaded {len(qa_pairs)} Q&A pairs")
        
        # Take a small sample for quick testing
        test_pairs = qa_pairs[:5] if len(qa_pairs) > 5 else qa_pairs
        
        # Initialize pipeline with fixes
        print("🚀 Initializing Enhanced Adaptive RAG Pipeline with fixes...")
        pipeline = AdaptiveRAGPipeline(test_pairs)
        
        # Test query processing
        test_queries = [
            "What is the purpose of a preamp?",
            "How does tube amplifier distortion work?",
            "What impedance should I use for my speakers?"
        ]
        
        print("\n🧪 Testing query processing...")
        for i, query in enumerate(test_queries):
            print(f"\n--- Test Query {i+1}: {query}")
            
            try:
                result = pipeline.process_query(query)
                
                print(f"✅ Query processed successfully")
                print(f"   - Strategy: {result['strategy_used']}")
                print(f"   - Retrieved items: {len(result['retrieved_items'])}")
                print(f"   - Context quality: {result['context_quality']:.3f}")
                print(f"   - Query type: {result['analysis'].query_type}")
                
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
                return False
        
        print("\n✅ All tests passed! Pipeline fixes working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_enhanced_pipeline_fixes()
    if success:
        print("\n🎉 Enhanced Adaptive RAG fixes validated successfully!")
        print("Ready for full evaluation run.")
    else:
        print("\n💥 Pipeline fixes still have issues - need more debugging.")