# Dual RAG AutoRAG Pipeline Architecture

## Overview

The Dual RAG AutoRAG pipeline transforms technical documents into high-quality training datasets through scientific comparison of two RAG approaches. The system evaluates both Standard and Adaptive RAG methods to automatically determine the optimal approach for deployment and training data generation.

## Enhanced Pipeline Flow

```
PDF → Q&A Generation (3×3 Matrix) → Selection → Dual Vector Stores → Parallel RAG Evaluation → A/B Comparison → Winner-Based Training Dataset
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

### Stage 4: Dual Vector Store Construction
- **Input**: Selected Q&A pairs
- **Process**: 
  - **Standard RAG**: Answer-only embeddings → FAISS indexing (GPU/CPU)
  - **Adaptive RAG**: Combined Q+A embeddings → FAISS indexing (GPU/CPU)
- **Output**: 4 FAISS indices (CPU/GPU × Standard/Adaptive)

### Stage 5: Parallel RAG Evaluation
- **Input**: Test questions + both vector stores
- **Process**: 
  - **Standard RAG**: Traditional retrieval and generation
  - **Adaptive RAG**: Context-aware retrieval and generation
- **Output**: Separate performance metrics for each approach

### Stage 6: RAG Comparison Analysis
- **Input**: Both evaluation results
- **Process**: Statistical comparison across quality, speed, and domain metrics
- **Output**: Winner determination and deployment recommendations

### Stage 7: Winner-Based Training Dataset Generation
- **Input**: Comparison analysis results
- **Process**: Use best-performing approach's results for training data
- **Output**: Optimized training dataset with approach selection rationale

## Key Design Principles

### Dual RAG Architecture
- **Scientific A/B Testing**: Parallel evaluation of Standard vs Adaptive approaches
- **Empirical Decision Making**: Winner selection based on quantitative metrics
- **Approach Specialization**: Different strategies optimized for speed vs quality
- **Automatic Optimization**: Self-selecting best approach for training data

### GPU-First Architecture  
- 4 FAISS GPU indices for comprehensive evaluation
- PyTorch with CUDA acceleration for both approaches
- Optional 8-bit quantization for memory efficiency
- Parallel processing for dual evaluation

### Enhanced Retrieval Strategies
- **Standard RAG**: Answer-only embeddings (speed-optimized)
- **Adaptive RAG**: Combined Q+A embeddings (quality-optimized)
- **Hybrid Search**: Dense + BM25 sparse matching in both approaches
- **Context Gating**: Confidence-based context selection

### Evidence-Based Quality Assessment
- Multi-dimensional comparison (quality, speed, domain relevance)
- Statistical significance testing between approaches
- Uncertainty detection and confidence calibration
- Performance vs resource trade-off analysis

### Domain Specialization with Comparison
- Configurable domain vocabulary and evaluation criteria
- Template-based prompt generation for both approaches
- Comparative domain-specific quality metrics
- Dual approach deployment recommendations

## Technical Stack

- **ML**: PyTorch 2.1+, Transformers 4.42+, Llama-3-8B-Instruct
- **Vector Search**: FAISS (GPU/CPU dual approach), Sentence-Transformers
- **Evaluation**: BERT-Score, comparative analysis, domain metrics
- **Analysis**: Statistical comparison framework, A/B testing tools
- **Infrastructure**: Poetry, GitHub Actions, L40S GPU, dual pipeline orchestration

## RAG Approach Comparison

### Standard RAG (Traditional)
- **Embedding Strategy**: Answer text only
- **Speed**: ⚡ Faster (single embedding lookup)
- **Memory**: Lower resource usage
- **Best For**: High-volume, simple queries
- **Use Cases**: FAQ systems, direct lookup scenarios

### Adaptive RAG (Enhanced)
- **Embedding Strategy**: Combined Question + Answer text
- **Quality**: 🎯 Better semantic understanding
- **Context**: Richer question-answer relationships
- **Best For**: Complex domain-specific queries
- **Use Cases**: Technical documentation, expert systems

### Automatic Selection
The pipeline automatically determines which approach performs better through:
1. **Parallel Evaluation**: Both approaches tested on identical data
2. **Multi-Metric Comparison**: Quality, speed, domain relevance analysis
3. **Winner Selection**: Empirical determination of best approach
4. **Training Data Generation**: Uses results from winning approach