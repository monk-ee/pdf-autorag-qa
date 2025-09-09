#!/usr/bin/env python3
"""
Test FAISS-GPU installation and functionality
Run this BEFORE the expensive pipeline to verify everything works
"""

import sys
import torch
import numpy as np

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import faiss
        print(f"✅ FAISS imported successfully (version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'})")
        return faiss
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return None

def test_pytorch_gpu():
    """Test PyTorch GPU detection"""
    print("\n🧪 Testing PyTorch GPU detection...")
    
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    print(f"✅ PyTorch CUDA available: {cuda_available}")
    print(f"✅ PyTorch device count: {device_count}")
    
    if cuda_available and device_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU name: {gpu_name}")
        print(f"✅ GPU memory: {memory:.1f}GB")
        return True
    else:
        print("❌ No GPU detected by PyTorch")
        return False

def test_faiss_gpu_detection(faiss):
    """Test FAISS GPU detection"""
    print("\n🧪 Testing FAISS GPU detection...")
    
    if not hasattr(faiss, 'get_num_gpus'):
        print("❌ faiss.get_num_gpus() not available")
        return False
    
    gpu_count = faiss.get_num_gpus()
    print(f"✅ FAISS GPU count: {gpu_count}")
    
    if gpu_count > 0:
        print("✅ FAISS can see GPUs")
        return True
    else:
        print("❌ FAISS cannot see any GPUs")
        return False

def test_faiss_gpu_classes(faiss):
    """Test FAISS GPU classes"""
    print("\n🧪 Testing FAISS GPU classes...")
    
    classes_to_test = [
        'StandardGpuResources',
        'index_cpu_to_gpu',
        'index_gpu_to_cpu'
    ]
    
    all_available = True
    for class_name in classes_to_test:
        if hasattr(faiss, class_name):
            print(f"✅ {class_name} available")
        else:
            print(f"❌ {class_name} NOT available")
            all_available = False
    
    return all_available

def test_faiss_gpu_operations(faiss):
    """Test actual FAISS GPU operations"""
    print("\n🧪 Testing FAISS GPU operations...")
    
    try:
        # Create a simple CPU index
        dimension = 64
        nb = 100
        data = np.random.random((nb, dimension)).astype('float32')
        
        print(f"Created test data: {data.shape}")
        
        # Build CPU index
        cpu_index = faiss.IndexFlatIP(dimension)
        cpu_index.add(data)
        print(f"✅ CPU index created with {cpu_index.ntotal} vectors")
        
        # Try to move to GPU
        if hasattr(faiss, 'StandardGpuResources'):
            print("Attempting GPU operations...")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"✅ GPU index created successfully")
            
            # Test search
            query = np.random.random((5, dimension)).astype('float32')
            scores, indices = gpu_index.search(query, 10)
            print(f"✅ GPU search completed: {scores.shape}")
            
            # Convert back to CPU
            cpu_from_gpu = faiss.index_gpu_to_cpu(gpu_index)
            print(f"✅ GPU->CPU conversion successful")
            
            return True
        else:
            print("❌ StandardGpuResources not available")
            return False
            
    except Exception as e:
        print(f"❌ FAISS GPU operations failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FAISS-GPU FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test 1: Basic imports
    faiss = test_basic_imports()
    if not faiss:
        print("\n💥 FAISS import failed - aborting tests")
        return 1
    
    # Test 2: PyTorch GPU
    pytorch_gpu = test_pytorch_gpu()
    
    # Test 3: FAISS GPU detection
    faiss_gpu_detection = test_faiss_gpu_detection(faiss)
    
    # Test 4: FAISS GPU classes
    faiss_gpu_classes = test_faiss_gpu_classes(faiss)
    
    # Test 5: FAISS GPU operations (if classes available)
    faiss_gpu_operations = False
    if faiss_gpu_classes:
        faiss_gpu_operations = test_faiss_gpu_operations(faiss)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    print(f"PyTorch GPU:        {'✅ PASS' if pytorch_gpu else '❌ FAIL'}")
    print(f"FAISS GPU Detection: {'✅ PASS' if faiss_gpu_detection else '❌ FAIL'}")
    print(f"FAISS GPU Classes:   {'✅ PASS' if faiss_gpu_classes else '❌ FAIL'}")
    print(f"FAISS GPU Operations: {'✅ PASS' if faiss_gpu_operations else '❌ FAIL'}")
    
    if all([pytorch_gpu, faiss_gpu_classes, faiss_gpu_operations]):
        print("\n🎉 ALL TESTS PASSED - FAISS-GPU is working!")
        print("✅ Safe to run expensive pipeline")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED - FAISS-GPU has issues")
        print("❌ DO NOT run expensive pipeline")
        return 1

if __name__ == '__main__':
    exit(main())