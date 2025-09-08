# Vector Store Construction (qa_faiss_builder.py)

## Purpose
Builds GPU-accelerated FAISS indices from Q&A pairs to enable fast semantic search for RAG (Retrieval-Augmented Generation).

## Why FAISS?
FAISS (Facebook AI Similarity Search) provides millisecond-scale vector search across millions of embeddings, with GPU acceleration for 10-100x speedup over CPU-only solutions.

## Vector Store Architecture

### Embedding Strategy
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Text Representation**: Combines question + answer for richer context
- **Normalization**: L2 normalization for cosine similarity search

### Index Construction
1. **Text Preprocessing**: Clean and tokenize Q&A pairs
2. **Batch Embedding**: Generate vectors in GPU-optimized batches
3. **Index Building**: Create FAISS flat index for exact search
4. **GPU Optimization**: Transfer index to GPU memory if available

## Technical Implementation

### Dual Index Strategy
```python
# Build both CPU and GPU indices for flexibility
cpu_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
gpu_index = faiss.GpuIndexFlatIP(gpu_resource, dimension)  # GPU acceleration
```

### Memory Management
- **Streaming Processing**: Handles large datasets without OOM
- **Batch Size Optimization**: Balances speed vs memory usage
- **GPU Memory Monitoring**: Automatic fallback to CPU if needed

## Core Components

```python
class QAFAISSBuilder:
    - load_qa_pairs()        # Read selected Q&A JSON
    - generate_embeddings()  # Batch embedding generation
    - build_indices()        # Create CPU + GPU FAISS indices
    - save_artifacts()       # Persist indices and metadata
```

### Embedding Process
1. **Text Preparation**: Format as "Q: {question} A: {answer}"
2. **Tokenization**: SentenceTransformer preprocessing
3. **Batch Inference**: GPU-accelerated embedding generation
4. **Normalization**: L2 norm for cosine similarity

## Configuration Options

### Performance Tuning
- `--batch-size`: Embedding batch size (default: 32)
- `--embedding-model`: SentenceTransformer model choice
- `--gpu-memory-fraction`: GPU memory allocation limit
- `--index-type`: FAISS index variant (Flat, IVF, HNSW)

### Output Formats
- **CPU Index**: `qa_faiss_index_cpu.bin` (universal compatibility)
- **GPU Index**: `qa_faiss_index_gpu.bin` (CUDA-accelerated)
- **Metadata**: `qa_metadata.json` (question mapping, quality scores)

## Index Performance

### Search Speed Comparison
- **CPU (Flat)**: ~1-10ms per query (50 Q&A pairs)
- **GPU (Flat)**: ~0.1-1ms per query (same dataset)
- **Scalability**: GPU advantage increases with dataset size

### Memory Requirements
- **Embeddings**: ~1.5KB per Q&A pair (384 dims × 4 bytes)
- **CPU Index**: Same as embeddings (in-memory)
- **GPU Index**: Additional GPU VRAM allocation

## Quality Assurance

### Embedding Quality Validation
- **Similarity Sanity Checks**: Verify related questions cluster together
- **Outlier Detection**: Identify potentially corrupted embeddings
- **Semantic Coherence**: Test question-answer alignment

### Index Integrity
- **Roundtrip Testing**: Query → retrieve → verify original pairs
- **Performance Benchmarking**: Measure search latency across index sizes
- **Cross-Platform Validation**: Ensure CPU/GPU indices return same results

## Integration Points

### Upstream Dependencies
- Requires output from `qa_pair_selector.py`
- Uses selected Q&A pairs with quality metadata
- Expects standardized JSON format

### Downstream Usage
- Consumed by `qa_autorag_evaluator.py` for RAG evaluation
- Enables fast context retrieval during inference
- Supports hybrid search strategies (dense + sparse)

## Advanced Features

### Hybrid Search Support
- **Dense Component**: FAISS semantic similarity
- **Sparse Component**: BM25 keyword matching integration
- **Score Fusion**: Weighted combination for optimal relevance

### Dynamic Updates
- **Incremental Indexing**: Add new Q&A pairs without full rebuild
- **Version Management**: Maintain multiple index versions
- **A/B Testing**: Compare different embedding strategies

## Use Cases
- RAG system knowledge bases
- Question answering over technical documents
- Semantic search for educational content
- Research into embedding model performance