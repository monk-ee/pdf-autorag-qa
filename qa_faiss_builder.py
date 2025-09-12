#!/usr/bin/env python3
"""
Simple Q&A FAISS Vector Store Builder for Standard RAG
Creates a single optimized FAISS index from curated Q&A pairs
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_qa_vector_store(qa_pairs_file: Path, 
                         output_dir: Path,
                         embedding_model: str = 'all-MiniLM-L6-v2') -> None:
    """
    Build FAISS vector store from Q&A pairs for Standard RAG.
    
    Args:
        qa_pairs_file: JSON file with selected Q&A pairs
        output_dir: Directory to save FAISS index and metadata
        embedding_model: SentenceTransformer model name
    
    Creates:
        - qa_faiss_index.bin (FAISS vector index)
        - qa_metadata.json (Q&A pairs metadata)
        - model_info.json (Build configuration)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🏗️ Building Q&A Vector Store")
    print(f"📁 Input: {qa_pairs_file}")
    print(f"📦 Output: {output_dir}")
    print(f"🧠 Model: {embedding_model}")
    
    # Load Q&A pairs
    print("\n📚 Loading Q&A pairs...")
    with open(qa_pairs_file, 'r') as f:
        qa_pairs = json.load(f)
    
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs")
    
    if len(qa_pairs) == 0:
        raise ValueError("No Q&A pairs found in input file")
    
    # Initialize embedding model
    print(f"\n🧠 Loading embedding model: {embedding_model}")
    device = 'cuda' if hasattr(faiss, 'StandardGpuResources') else 'cpu'
    model = SentenceTransformer(embedding_model, device=device)
    print(f"✅ Model loaded on {device}")
    
    # Generate embeddings for answers (standard RAG approach)
    print("\n🔍 Generating embeddings...")
    # Handle both formats: new (instruction/output) and old (question/answer)
    answers = [pair.get('output') or pair.get('answer') for pair in qa_pairs]
    
    start_time = time.time()
    embeddings = model.encode(
        answers,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embedding_time = time.time() - start_time
    
    print(f"✅ Generated {len(embeddings)} embeddings in {embedding_time:.1f}s")
    print(f"📊 Embedding shape: {embeddings.shape}")
    
    # Ensure float32 for FAISS compatibility
    if embeddings.dtype != np.float32:
        print(f"🔧 Converting embeddings from {embeddings.dtype} to float32...")
        embeddings = embeddings.astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    print("🔧 Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS indices (CPU and GPU versions)
    print("\n🏗️ Building FAISS indices...")
    dimension = embeddings.shape[1]
    
    # CPU version (always works)
    cpu_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
    cpu_index.add(embeddings)
    print(f"✅ CPU FAISS index built with {cpu_index.ntotal} vectors")
    
    # Save CPU index
    cpu_index_path = output_dir / 'qa_faiss_index_cpu.bin'
    faiss.write_index(cpu_index, str(cpu_index_path))
    print(f"💾 CPU FAISS index saved: {cpu_index_path}")
    
    # GPU version (if GPU available)
    gpu_index_path = output_dir / 'qa_faiss_index_gpu.bin'
    if device == 'cuda' and faiss.get_num_gpus() > 0:
        try:
            print("🚀 Creating GPU FAISS index...")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            # Convert back to CPU for saving
            gpu_index_cpu = faiss.index_gpu_to_cpu(gpu_index)
            faiss.write_index(gpu_index_cpu, str(gpu_index_path))
            print(f"💾 GPU FAISS index saved: {gpu_index_path}")
        except Exception as e:
            print(f"⚠️ GPU index creation failed: {e}")
            print("🔄 Copying CPU index as GPU fallback...")
            faiss.write_index(cpu_index, str(gpu_index_path))
    else:
        print("🔄 No GPU available, copying CPU index as GPU fallback...")
        faiss.write_index(cpu_index, str(gpu_index_path))
    
    # Default index (backwards compatibility)
    default_index_path = output_dir / 'qa_faiss_index.bin'
    faiss.write_index(cpu_index, str(default_index_path))
    print(f"💾 Default FAISS index saved: {default_index_path}")
    
    # Save metadata
    metadata_path = output_dir / 'qa_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")
    
    # Save model info
    model_info = {
        'embedding_model': embedding_model,
        'dimension': dimension,
        'num_vectors': len(embeddings),
        'index_type': 'IndexFlatIP',
        'device_used': device,
        'build_time_seconds': embedding_time,
        'normalized': True
    }
    
    model_info_path = output_dir / 'model_info.json'
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"💾 Model info saved: {model_info_path}")
    
    print(f"\n🎉 Vector store build complete!")
    print(f"📊 Summary:")
    print(f"   - Q&A pairs: {len(qa_pairs)}")
    print(f"   - Embedding dimension: {dimension}")
    print(f"   - Index type: Flat IP (cosine similarity)")
    print(f"   - Build time: {embedding_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Build Q&A FAISS vector store for Standard RAG')
    parser.add_argument('--qa-pairs-file', type=str, required=True,
                        help='Path to Q&A pairs JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for FAISS index and metadata')
    parser.add_argument('--embedding-model', type=str, 
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        build_qa_vector_store(
            qa_pairs_file=Path(args.qa_pairs_file),
            output_dir=Path(args.output_dir),
            embedding_model=args.embedding_model
        )
        print("✅ Success!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()