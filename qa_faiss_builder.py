#!/usr/bin/env python3
"""
Q&A-Based FAISS Vector Store Builder for AutoRAG Pipeline

Builds FAISS index from SELECTED Q&A pairs, not raw PDF text.
This is the correct approach - we use our curated Q&A pairs as the knowledge base.
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Handle FAISS imports with graceful fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu faiss-gpu")


def build_qa_vector_store(qa_pairs_file: Path, 
                         output_dir: Path,
                         embedding_model: str = 'all-MiniLM-L6-v2') -> None:
    """
    Build FAISS vector store from selected Q&A pairs.
    
    Args:
        qa_pairs_file: JSON file with selected Q&A pairs
        output_dir: Directory to save FAISS index and metadata
        embedding_model: SentenceTransformer model name
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📥 Loading Q&A pairs from: {qa_pairs_file}')
    with open(qa_pairs_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f'📊 Loaded {len(qa_pairs)} Q&A pairs for vector store')
    
    # Load embedding model
    print(f'🔧 Loading embedding model: {embedding_model}')
    model = SentenceTransformer(embedding_model)
    
    # Prepare texts for embedding (we'll embed the answers as our knowledge base)
    texts = []
    qa_metadata = []
    
    for i, pair in enumerate(qa_pairs):
        # Use answer as the knowledge base content (handle both HF and QA formats)
        answer_text = pair.get('output', pair.get('answer', ''))
        question_text = pair.get('instruction', pair.get('question', ''))
        
        if len(answer_text.strip()) < 10:  # Skip very short answers
            print(f'   ⚠️ Skipping pair {i}: answer too short ({len(answer_text)} chars)')
            continue
            
        texts.append(answer_text)
        qa_metadata.append({
            'id': i,
            'question': question_text,
            'answer': answer_text,
            'source_info': pair.get('source_info', {}),
            'quality_score': pair.get('quality_score', 0.0),
            'matrix_combination': pair.get('matrix_combination', ''),
            'metadata': pair.get('metadata', {})
        })
    
    print(f'📊 Building vector store with {len(texts)} answer embeddings')
    
    # Generate embeddings
    print('🔍 Generating embeddings...')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f'✅ Generated embeddings shape: {embeddings.shape}')
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings generated - all answers may be too short or empty")
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product similarity (good for sentence embeddings)
    cpu_index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to CPU index first
    cpu_index.add(embeddings.astype(np.float32))
    
    print(f'🏗️ Built FAISS index with {cpu_index.ntotal} vectors')
    
    # Save CPU index for compatibility
    cpu_index_path = output_dir / 'qa_faiss_index_cpu.bin'
    faiss.write_index(cpu_index, str(cpu_index_path))
    print(f'💾 Saved CPU FAISS index: {cpu_index_path}')
    
    # Initialize model info
    model_info = {
        'embedding_model': embedding_model,
        'dimension': int(dimension),
        'total_vectors': int(cpu_index.ntotal),
        'index_type': 'IndexFlatIP',
        'similarity_metric': 'cosine',
        'normalized': True,
        'gpu_available': False  # Will be updated if GPU index created
    }
    
    # Try to create GPU index if available
    try:
        if faiss.get_num_gpus() > 0:
            print('🚀 GPU detected - creating GPU-optimized FAISS index')
            
            # Move to GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            # Save GPU index
            gpu_index_path = output_dir / 'qa_faiss_index_gpu.bin'
            # Convert back to CPU for saving (GPU indices need to be serialized differently)
            cpu_from_gpu = faiss.index_gpu_to_cpu(gpu_index)
            faiss.write_index(cpu_from_gpu, str(gpu_index_path))
            print(f'💾 Saved GPU FAISS index: {gpu_index_path}')
            
            # Update model info to indicate GPU availability
            model_info['gpu_available'] = True
            model_info['gpu_index_path'] = str(gpu_index_path)
        else:
            print('⚠️ No GPU detected - using CPU-only index')
            model_info['gpu_available'] = False
    except Exception as e:
        print(f'⚠️ GPU index creation failed: {e} - falling back to CPU index')
        model_info['gpu_available'] = False
    
    # Save metadata
    metadata_path = output_dir / 'qa_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(qa_metadata, f, indent=2, ensure_ascii=False)
    print(f'💾 Saved metadata: {metadata_path}')
    
    info_path = output_dir / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f'💾 Saved model info: {info_path}')
    
    print('✅ Q&A vector store built successfully!')
    print(f'   - CPU FAISS Index: {cpu_index_path}')
    print(f'   - Metadata: {metadata_path}')
    print(f'   - Model Info: {info_path}')


def main():
    parser = argparse.ArgumentParser(description='Build Q&A-based FAISS vector store')
    parser.add_argument('--qa-pairs-file', type=Path, required=True,
                       help='JSON file with selected Q&A pairs')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for FAISS index')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model name')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        build_qa_vector_store(
            qa_pairs_file=args.qa_pairs_file,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model
        )
    except Exception as e:
        print(f'❌ Error building Q&A vector store: {e}')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())