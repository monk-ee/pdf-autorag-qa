# AutoRAG Pipeline Architecture

## Overview

The AutoRAG pipeline transforms technical documents into high-quality training datasets through a series of GPU-accelerated ML operations. Each component is designed to be modular, debuggable, and scalable.

## Pipeline Flow

```
PDF → Q&A Generation (3×3 Matrix) → Selection → Vector Store → RAG Evaluation → Training Dataset
```

### Stage 1: Document Processing
- **Input**: Technical PDF (guitar amp manual)
- **Process**: Text extraction, chunking, prompt generation
- **Output**: Raw text chunks ready for Q&A generation

### Stage 2: Q&A Generation Matrix (3×3)
- **Difficulty Levels**: Basic, Intermediate, Advanced
- **Creativity Styles**: Conservative (0.3), Balanced (0.7), High Creativity (0.9)
- **Process**: 9 separate Llama-3 inference runs with different sampling parameters
- **Output**: ~500-1000 Q&A pairs across all combinations

### Stage 3: Quality Selection
- **Input**: All generated Q&A pairs with metadata
- **Process**: Semantic similarity scoring, quality filtering
- **Output**: Top-K highest quality pairs (default: 50)

### Stage 4: Vector Store Construction
- **Input**: Selected Q&A pairs
- **Process**: Sentence-Transformer embeddings → FAISS indexing (GPU/CPU)
- **Output**: Searchable knowledge base

### Stage 5: RAG Evaluation
- **Input**: Test questions + vector store
- **Process**: Base model vs RAG-enhanced response comparison
- **Output**: Performance metrics (BERT-Score, domain relevance, etc.)

### Stage 6: Training Dataset Generation
- **Input**: RAG evaluation results
- **Process**: Quality filtering based on evaluation metrics
- **Output**: Production-ready training dataset (JSONL format)

## Key Design Principles

### GPU-First Architecture
- FAISS GPU indexing for sub-millisecond retrieval
- PyTorch with CUDA acceleration
- Optional 8-bit quantization for memory efficiency

### Hybrid Retrieval
- **Dense**: Semantic similarity via sentence transformers
- **Sparse**: BM25 keyword matching
- **Combined**: Weighted fusion for optimal relevance

### Quality-Driven Selection
- Multi-metric evaluation (semantic, syntactic, domain-specific)
- Uncertainty detection for out-of-domain filtering
- Confidence-based context gating

### Domain Specialization
- Configurable domain vocabulary and evaluation criteria
- Template-based prompt generation
- Domain-specific quality metrics

## Technical Stack

- **ML**: PyTorch 2.1+, Transformers 4.42+, Llama-3-8B-Instruct
- **Vector Search**: FAISS (GPU/CPU), Sentence-Transformers
- **Evaluation**: BERT-Score, custom domain metrics
- **Infrastructure**: Poetry, GitHub Actions, L40S GPU