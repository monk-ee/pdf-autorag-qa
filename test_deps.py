#!/usr/bin/env python3
"""
Test script to verify all critical dependencies work locally
Run this before pushing to CI to avoid pipeline failures
"""

import sys
import traceback

def test_import(module_name, package_name=None):
    """Test importing a module and print results"""
    try:
        if package_name:
            exec(f"import {module_name}")
            print(f"✅ {package_name}: OK")
            # Get version if available
            try:
                version = eval(f"{module_name}.__version__")
                print(f"   Version: {version}")
            except:
                pass
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name}: OK")
    except ImportError as e:
        print(f"❌ {package_name or module_name}: FAILED")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name or module_name}: Import OK but error getting version")
        print(f"   Error: {e}")
    return True

def test_faiss():
    """Test FAISS specifically"""
    try:
        import faiss
        print("✅ faiss: OK")
        
        # Test if GPU is available
        try:
            if faiss.get_num_gpus() > 0:
                print(f"   GPU support: {faiss.get_num_gpus()} GPU(s) available")
            else:
                print("   GPU support: No GPUs detected (CPU only)")
        except:
            print("   GPU support: Could not check GPU status")
            
        # Test basic functionality
        import numpy as np
        dimension = 64
        index = faiss.IndexFlatL2(dimension)
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        print(f"   Basic functionality: Added {index.ntotal} vectors")
        return True
        
    except Exception as e:
        print(f"❌ faiss: FAILED")
        print(f"   Error: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers functionality"""
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers: Import OK")
        
        # Test loading a model (this is what usually fails)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ sentence_transformers: Model loading OK")
        
        # Test encoding
        sentences = ["This is a test sentence", "This is another test"]
        embeddings = model.encode(sentences)
        print(f"✅ sentence_transformers: Encoding OK ({embeddings.shape})")
        return True
        
    except Exception as e:
        print(f"❌ sentence_transformers: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all dependency tests"""
    print("🧪 Testing AutoRAG Pipeline Dependencies")
    print("=" * 50)
    
    # Core ML dependencies
    print("\n📦 Core ML Dependencies:")
    test_import("torch")
    test_import("transformers")
    test_import("accelerate")
    test_import("bitsandbytes")
    
    # Vector search and embeddings
    print("\n🔍 Vector Search & Embeddings:")
    faiss_ok = test_faiss()
    st_ok = test_sentence_transformers()
    
    # Evaluation and metrics  
    print("\n📊 Evaluation & Metrics:")
    test_import("bert_score")
    test_import("sklearn", "scikit-learn")
    test_import("rank_bm25")
    
    # Data processing
    print("\n📋 Data Processing:")
    test_import("pandas")
    test_import("numpy")
    test_import("tqdm")
    
    # PDF processing
    print("\n📄 PDF Processing:")
    test_import("PyPDF2")
    test_import("fitz", "PyMuPDF")
    
    # Additional dependencies
    print("\n🔧 Additional:")
    test_import("optimum")
    test_import("datasets")
    test_import("multiprocess")
    test_import("psutil")
    test_import("huggingface_hub")
    
    # Summary
    print("\n" + "=" * 50)
    if faiss_ok and st_ok:
        print("🎉 ALL CRITICAL DEPENDENCIES OK!")
        print("✅ Pipeline should work in CI")
        return 0
    else:
        print("💥 CRITICAL DEPENDENCIES FAILED!")
        print("❌ Fix these before running CI pipeline")
        return 1

if __name__ == "__main__":
    sys.exit(main())