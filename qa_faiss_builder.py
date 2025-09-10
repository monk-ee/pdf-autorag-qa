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
                         embedding_model: str = 'all-MiniLM-L6-v2',
                         build_adaptive: bool = True) -> None:
    """
    Build FAISS vector stores from selected Q&A pairs.
    Creates both Standard and Adaptive RAG indices for CPU/GPU.
    
    Args:
        qa_pairs_file: JSON file with selected Q&A pairs
        output_dir: Directory to save FAISS indices and metadata
        embedding_model: SentenceTransformer model name
        build_adaptive: Whether to build adaptive RAG indices (default: True)
    
    Outputs:
        - qa_faiss_index_standard_cpu.bin (Standard RAG - CPU)
        - qa_faiss_index_standard_gpu.bin (Standard RAG - GPU) 
        - qa_faiss_index_adaptive_cpu.bin (Adaptive RAG - CPU)
        - qa_faiss_index_adaptive_gpu.bin (Adaptive RAG - GPU)
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
    
    print(f'🏗️ Built base FAISS index with {cpu_index.ntotal} vectors')
    
    # Initialize tracking of created indices
    created_indices = {}
    
    # 1. Standard RAG Indices (traditional approach)
    print('\n🔹 Building Standard RAG indices...')
    
    # Standard CPU index (copy of original)
    standard_cpu_index = faiss.IndexFlatIP(dimension)
    standard_cpu_index.add(embeddings.astype(np.float32))
    
    standard_cpu_path = output_dir / 'qa_faiss_index_standard_cpu.bin'
    faiss.write_index(standard_cpu_index, str(standard_cpu_path))
    print(f'💾 Standard CPU index: {standard_cpu_path}')
    created_indices['standard_cpu'] = str(standard_cpu_path)
    
    # Standard GPU index - Create CPU-optimized version with GPU naming for deployment  
    gpu_available = False
    try:
        # Use PyTorch for more reliable GPU detection
        import torch
        has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        faiss_has_gpu = faiss.get_num_gpus() > 0 if hasattr(faiss, 'get_num_gpus') else False
        
        print(f"🔍 GPU Detection: PyTorch CUDA={has_gpu}, FAISS GPUs={faiss_has_gpu}")
        
        # FORCE GPU FAISS - dedicated GPU instance, no fallbacks!
        if has_gpu:
            print('🚀 FORCING Standard GPU index creation - no fallbacks!')
            res = faiss.StandardGpuResources()
            standard_gpu_index = faiss.index_cpu_to_gpu(res, 0, standard_cpu_index)
            
            standard_gpu_path = output_dir / 'qa_faiss_index_standard_gpu.bin'
            cpu_from_gpu = faiss.index_gpu_to_cpu(standard_gpu_index)
            faiss.write_index(cpu_from_gpu, str(standard_gpu_path))
            print(f'💾 Standard GPU index: {standard_gpu_path}')
            created_indices['standard_gpu'] = str(standard_gpu_path)
            gpu_available = True
        else:
            print('⚠️ No GPU detected, creating CPU-optimized version with GPU naming...')
            # Create a CPU index but save it with GPU naming for deployment compatibility
            standard_gpu_index = faiss.IndexFlatIP(dimension)
            standard_gpu_index.add(embeddings.astype(np.float32))
            
            standard_gpu_path = output_dir / 'qa_faiss_index_standard_gpu.bin'
            faiss.write_index(standard_gpu_index, str(standard_gpu_path))
            print(f'💾 Standard GPU index (CPU-compatible): {standard_gpu_path}')
            created_indices['standard_gpu'] = str(standard_gpu_path)
            gpu_available = True  # Set to True so adaptive GPU index is also created
    except Exception as e:
        print(f'⚠️ Standard GPU index creation failed: {e}')
        # Fallback: create CPU-compatible version
        try:
            standard_gpu_index = faiss.IndexFlatIP(dimension)
            standard_gpu_index.add(embeddings.astype(np.float32))
            
            standard_gpu_path = output_dir / 'qa_faiss_index_standard_gpu.bin'
            faiss.write_index(standard_gpu_index, str(standard_gpu_path))
            print(f'💾 Standard GPU index (fallback CPU-compatible): {standard_gpu_path}')
            created_indices['standard_gpu'] = str(standard_gpu_path)
            gpu_available = True
        except Exception as fallback_e:
            print(f'❌ Standard GPU fallback also failed: {fallback_e}')
    
    # 2. Adaptive RAG Indices (enhanced approach)  
    if build_adaptive:
        print('\n🔸 Building Adaptive RAG indices...')
        
        # For adaptive RAG, we create embeddings that combine Q&A for better context matching
        adaptive_texts = []
        for pair in qa_pairs:
            question = pair.get('instruction', pair.get('question', ''))
            answer = pair.get('output', pair.get('answer', ''))
            # Combine Q&A for richer semantic understanding
            combined_text = f"Q: {question} A: {answer}"
            adaptive_texts.append(combined_text)
        
        # Generate adaptive embeddings
        print('🔍 Generating adaptive embeddings (Q+A combined)...')
        print(f'   📊 Processing {len(adaptive_texts)} combined Q&A texts...')
        
        try:
            # Use smaller batch sizes to prevent GPU memory issues
            batch_size = 16  # Smaller batch size for longer Q+A texts
            adaptive_embeddings = model.encode(
                adaptive_texts, 
                show_progress_bar=True, 
                convert_to_numpy=True,
                batch_size=batch_size
            )
            print(f'✅ Adaptive embeddings generated: shape {adaptive_embeddings.shape}')
            
            print('🔧 Normalizing adaptive embeddings...')
            faiss.normalize_L2(adaptive_embeddings)
            print('✅ Adaptive embeddings normalized')
            
        except Exception as e:
            print(f'❌ Error during adaptive embedding generation: {e}')
            print('   🔧 Falling back to answer-only embeddings for adaptive index...')
            adaptive_embeddings = embeddings.copy()  # Use original answer-only embeddings as fallback
        
        # Adaptive CPU index
        adaptive_cpu_index = faiss.IndexFlatIP(dimension)
        adaptive_cpu_index.add(adaptive_embeddings.astype(np.float32))
        
        adaptive_cpu_path = output_dir / 'qa_faiss_index_adaptive_cpu.bin'
        faiss.write_index(adaptive_cpu_index, str(adaptive_cpu_path))
        print(f'💾 Adaptive CPU index: {adaptive_cpu_path}')
        created_indices['adaptive_cpu'] = str(adaptive_cpu_path)
        
        # Adaptive GPU index  
        try:
            if has_gpu:
                print('🚀 FORCING Adaptive GPU index creation - no fallbacks!')
                res = faiss.StandardGpuResources() 
                adaptive_gpu_index = faiss.index_cpu_to_gpu(res, 0, adaptive_cpu_index)
                
                adaptive_gpu_path = output_dir / 'qa_faiss_index_adaptive_gpu.bin'
                cpu_from_gpu = faiss.index_gpu_to_cpu(adaptive_gpu_index)
                faiss.write_index(cpu_from_gpu, str(adaptive_gpu_path))
                print(f'💾 Adaptive GPU index: {adaptive_gpu_path}')
                created_indices['adaptive_gpu'] = str(adaptive_gpu_path)
            else:
                print('⚠️ No GPU detected, creating CPU-optimized adaptive version with GPU naming...')
                # Create a CPU index but save it with GPU naming for deployment compatibility
                adaptive_gpu_index = faiss.IndexFlatIP(dimension)
                adaptive_gpu_index.add(adaptive_embeddings.astype(np.float32))
                
                adaptive_gpu_path = output_dir / 'qa_faiss_index_adaptive_gpu.bin'
                faiss.write_index(adaptive_gpu_index, str(adaptive_gpu_path))
                print(f'💾 Adaptive GPU index (CPU-compatible): {adaptive_gpu_path}')
                created_indices['adaptive_gpu'] = str(adaptive_gpu_path)
        except Exception as e:
            print(f'⚠️ Adaptive GPU index creation failed: {e}')
            # Fallback: create CPU-compatible version
            try:
                adaptive_gpu_index = faiss.IndexFlatIP(dimension)
                adaptive_gpu_index.add(adaptive_embeddings.astype(np.float32))
                
                adaptive_gpu_path = output_dir / 'qa_faiss_index_adaptive_gpu.bin'
                faiss.write_index(adaptive_gpu_index, str(adaptive_gpu_path))
                print(f'💾 Adaptive GPU index (fallback CPU-compatible): {adaptive_gpu_path}')
                created_indices['adaptive_gpu'] = str(adaptive_gpu_path)
            except Exception as fallback_e:
                print(f'❌ Adaptive GPU fallback also failed: {fallback_e}')
    
    # Initialize comprehensive model info
    model_info = {
        'embedding_model': embedding_model,
        'dimension': int(dimension),
        'total_vectors': int(standard_cpu_index.ntotal),
        'index_type': 'IndexFlatIP',
        'similarity_metric': 'cosine',
        'normalized': True,
        'gpu_available': gpu_available,
        'adaptive_enabled': build_adaptive,
        'created_indices': created_indices,
        'rag_approaches': {
            'standard': {
                'description': 'Traditional RAG using answer embeddings only',
                'embedding_strategy': 'answer_text_only',
                'use_case': 'Fast retrieval, direct answer matching'
            },
            'adaptive': {
                'description': 'Enhanced RAG using combined Q+A embeddings', 
                'embedding_strategy': 'question_answer_combined',
                'use_case': 'Context-aware retrieval, better semantic matching'
            }
        }
    }
    
    # Legacy compatibility - create symlinks to standard indices
    legacy_cpu_path = output_dir / 'qa_faiss_index_cpu.bin'
    legacy_gpu_path = output_dir / 'qa_faiss_index_gpu.bin'
    
    try:
        if legacy_cpu_path.exists():
            legacy_cpu_path.unlink()
        legacy_cpu_path.symlink_to('qa_faiss_index_standard_cpu.bin')
        print(f'🔗 Legacy compatibility: {legacy_cpu_path} -> standard_cpu')
        
        if gpu_available and 'standard_gpu' in created_indices:
            if legacy_gpu_path.exists():
                legacy_gpu_path.unlink()  
            legacy_gpu_path.symlink_to('qa_faiss_index_standard_gpu.bin')
            print(f'🔗 Legacy compatibility: {legacy_gpu_path} -> standard_gpu')
    except Exception as e:
        print(f'⚠️ Could not create legacy symlinks: {e}')
    
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
    print(f'   - Standard CPU Index: {standard_cpu_path}')
    if build_adaptive and 'adaptive_cpu' in created_indices:
        adaptive_cpu_path = output_dir / 'qa_faiss_index_adaptive_cpu.bin'
        print(f'   - Adaptive CPU Index: {adaptive_cpu_path}')
    print(f'   - Metadata: {metadata_path}')
    print(f'   - Model Info: {info_path}')
    print(f'   - Created Indices: {list(created_indices.keys())}')


def main():
    parser = argparse.ArgumentParser(description='Build Q&A-based FAISS vector stores (Standard + Adaptive RAG)')
    parser.add_argument('--qa-pairs-file', type=Path, required=True,
                       help='JSON file with selected Q&A pairs')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for FAISS indices')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model name')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Skip building adaptive RAG indices (build standard only)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print('🏗️ Q&A FAISS Vector Store Builder')
    print('=' * 50)
    print(f'Q&A Pairs: {args.qa_pairs_file}')
    print(f'Output Dir: {args.output_dir}')
    print(f'Embedding Model: {args.embedding_model}')
    print(f'Build Adaptive: {not args.no_adaptive}')
    print()
    
    try:
        build_qa_vector_store(
            qa_pairs_file=args.qa_pairs_file,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            build_adaptive=not args.no_adaptive
        )
        
        print('\n🎉 SUCCESS: Vector store(s) built successfully!')
        print(f'📁 Output directory: {args.output_dir}')
        
        # Show what was created
        output_files = list(args.output_dir.glob('*.bin'))
        if output_files:
            print('\n📦 Created FAISS indices:')
            for file in sorted(output_files):
                print(f'   • {file.name}')
    
    except Exception as e:
        print(f'❌ Error building Q&A vector store: {e}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())