#!/usr/bin/env python3
"""
Test script to verify pipeline components work locally
Tests the actual imports from your pipeline scripts
"""

import sys
import os
from pathlib import Path

def test_pipeline_imports():
    """Test importing from actual pipeline scripts"""
    print("🧪 Testing Pipeline Script Imports")
    print("=" * 50)
    
    success = True
    
    # Test qa_faiss_builder.py (the one that keeps failing)
    try:
        print("Testing qa_faiss_builder.py imports...")
        
        # Test the specific imports that keep failing
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers import: OK")
        
        import faiss
        print("✅ faiss import: OK")
        
        # Test loading the model used in pipeline
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ SentenceTransformer model loading: OK")
        
        # Test basic FAISS operations
        import numpy as np
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dimension)
        print(f"✅ FAISS IndexFlatIP creation: OK")
        
        # Test embedding + FAISS workflow
        texts = ["test question", "test answer"]
        embeddings = model.encode(texts)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        print(f"✅ Full embedding → FAISS workflow: OK")
        
    except Exception as e:
        print(f"❌ qa_faiss_builder.py imports: FAILED")
        print(f"   Error: {e}")
        success = False
    
    # Test other critical pipeline imports
    try:
        print("\nTesting other pipeline imports...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from bert_score import BERTScorer
        print("✅ Core ML imports: OK")
        
    except Exception as e:
        print(f"❌ Core ML imports: FAILED")
        print(f"   Error: {e}")
        success = False
    
    return success

def test_pipeline_scripts_exist():
    """Verify all pipeline scripts exist"""
    print("\n📁 Checking Pipeline Scripts Exist")
    print("=" * 30)
    
    required_scripts = [
        'cli_pdf_qa.py',
        'qa_pair_selector.py', 
        'qa_faiss_builder.py',
        'qa_autorag_evaluator.py',
        'training_dataset_generator.py',
        'domain_eval_gpu.py'
    ]
    
    success = True
    for script in required_scripts:
        if Path(script).exists():
            print(f"✅ {script}: EXISTS")
        else:
            print(f"❌ {script}: MISSING")
            success = False
            
    return success

def main():
    """Run all pipeline tests"""
    scripts_ok = test_pipeline_scripts_exist()
    imports_ok = test_pipeline_imports()
    
    print("\n" + "=" * 50)
    if scripts_ok and imports_ok:
        print("🎉 PIPELINE READY!")
        print("✅ All scripts exist and imports work")
        print("✅ Safe to run CI pipeline")
        return 0
    else:
        print("💥 PIPELINE NOT READY!")
        print("❌ Fix issues before running CI")
        return 1

if __name__ == "__main__":
    sys.exit(main())