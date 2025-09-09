# Dual Vector Store Construction (qa_faiss_builder.py)

## Purpose
Builds **4 GPU-accelerated FAISS indices** from Q&A pairs using both Standard and Adaptive RAG approaches for comprehensive evaluation and optimal deployment strategy selection.

## Why Dual RAG?
The system builds two complementary approaches:
- **Standard RAG**: Traditional answer-only embeddings (speed-optimized)
- **Adaptive RAG**: Combined Q+A embeddings (quality-optimized)

This enables scientific A/B testing to determine the best approach for specific use cases.

## Dual Vector Store Architecture

### Standard RAG Embedding Strategy
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Text Representation**: Answer text only
- **Optimization**: Speed and memory efficiency
- **Use Case**: High-volume, direct lookup scenarios

### Adaptive RAG Embedding Strategy  
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Text Representation**: Combined "Q: {question} A: {answer}" format
- **Optimization**: Semantic understanding and context awareness
- **Use Case**: Complex domain-specific queries

### Dual Index Construction Process
1. **Standard RAG Process**:
   - Extract answer texts from Q&A pairs
   - Generate answer-only embeddings
   - Build Standard CPU + GPU FAISS indices
   
2. **Adaptive RAG Process**:
   - Combine questions and answers ("Q: ... A: ...")
   - Generate combined embeddings for richer context
   - Build Adaptive CPU + GPU FAISS indices

3. **Legacy Compatibility**: Create symlinks for backward compatibility

## Technical Implementation

### Four Index Strategy
```python
# Standard RAG indices (answer-only embeddings)
standard_cpu_index = faiss.IndexFlatIP(dimension)
standard_gpu_index = faiss.index_cpu_to_gpu(res, 0, standard_cpu_index)

# Adaptive RAG indices (Q+A combined embeddings)  
adaptive_cpu_index = faiss.IndexFlatIP(dimension)
adaptive_gpu_index = faiss.index_cpu_to_gpu(res, 0, adaptive_cpu_index)
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
- `--embedding-model`: SentenceTransformer model choice (default: all-MiniLM-L6-v2)
- `--no-adaptive`: Skip building adaptive indices (build standard only)
- `--verbose`: Enable detailed logging and progress tracking

### Output Formats (4 Indices Generated)
- **Standard CPU**: `qa_faiss_index_standard_cpu.bin` (traditional approach, CPU)
- **Standard GPU**: `qa_faiss_index_standard_gpu.bin` (traditional approach, GPU)
- **Adaptive CPU**: `qa_faiss_index_adaptive_cpu.bin` (enhanced approach, CPU)
- **Adaptive GPU**: `qa_faiss_index_adaptive_gpu.bin` (enhanced approach, GPU)
- **Legacy Links**: `qa_faiss_index_cpu.bin` → standard_cpu, `qa_faiss_index_gpu.bin` → standard_gpu
- **Metadata**: `qa_metadata.json` (comprehensive comparison and approach info)

## Index Performance

### Search Speed Comparison
- **Standard CPU**: ~1-10ms per query (direct answer lookup)
- **Standard GPU**: ~0.1-1ms per query (GPU-accelerated)
- **Adaptive CPU**: ~1-10ms per query (Q+A context understanding)
- **Adaptive GPU**: ~0.1-1ms per query (GPU-accelerated with richer context)

### Memory Requirements (×2 for Dual Approach)
- **Standard Embeddings**: ~1.5KB per Q&A pair (384 dims × 4 bytes, answer-only)
- **Adaptive Embeddings**: ~1.5KB per Q&A pair (384 dims × 4 bytes, Q+A combined)
- **Total Storage**: ~3KB per Q&A pair (both approaches)
- **GPU VRAM**: Additional allocation for GPU indices

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
- Consumed by `qa_autorag_evaluator.py` for parallel RAG evaluation
- Enables comparative performance testing (Standard vs Adaptive)
- Supports dual hybrid search strategies (dense + sparse for both approaches)
- Used by `rag_comparison_analyzer.py` for A/B testing analysis

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

### Standard RAG Indices
- High-volume FAQ systems
- Direct answer lookup scenarios  
- Speed-critical applications
- Resource-constrained deployments

### Adaptive RAG Indices
- Complex technical documentation
- Domain-specific expert systems
- Context-aware question answering
- Quality-critical applications

### Comparative Analysis
- A/B testing different RAG approaches
- Performance optimization research
- Deployment strategy selection
- Scientific evaluation of retrieval methods