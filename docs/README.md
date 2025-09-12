# AutoRAG Pipeline Documentation

## Component Overview

This documentation explains each component of the AutoRAG pipeline in technical detail. Each component is designed to be modular, debuggable, and production-ready.

## Pipeline Components

### 1. [Q&A Generation Engine](cli_pdf_qa.md)
**Purpose**: Extract Q&A pairs from PDFs using GPU-accelerated LLMs  
**Key Features**: 3×3 difficulty/creativity matrix, batch processing, quality filtering  
**Output**: 500-1000 Q&A pairs across 9 parameter combinations

### 2. [Quality-Based Selection](qa_pair_selector.md)
**Purpose**: Select highest-quality pairs using multi-metric evaluation  
**Key Features**: Semantic scoring, deduplication, diversity preservation  
**Output**: Top-K pairs (default: 50) with quality metadata

### 3. Vector Store Construction (qa_faiss_builder.py)
**Purpose**: Build GPU-accelerated FAISS indices for fast retrieval  
**Key Features**: Dual CPU/GPU indices, normalized embeddings, cosine similarity  
**Output**: Searchable knowledge base with sub-millisecond queries

### 4. RAG Performance Evaluation (qa_autorag_evaluator.py)
**Purpose**: Compare base vs RAG-enhanced model responses  
**Key Features**: BERT-Score evaluation, semantic similarity, quality metrics  
**Output**: Performance metrics and improvement quantification

### 5. Training Dataset Generation (training_dataset_generator.py)
**Purpose**: Convert evaluation results to production training data  
**Key Features**: Performance-based filtering, quality scoring, format conversion  
**Output**: High-quality JSONL training dataset

### 6. [Domain Specificity Evaluation](domain_eval_gpu.md)
**Purpose**: Measure domain expertise vs general knowledge  
**Key Features**: Configurable domains, uncertainty analysis, technical term scoring  
**Output**: Domain specialization effectiveness metrics

## Available Components

All core pipeline components are implemented and ready for use. See individual documentation files linked above for detailed technical specifications and usage examples.

## Quick Navigation

- **Getting Started**: See main [README.md](../README.md)
- **Component Details**: Individual component documentation above
- **Configuration**: Each component doc includes configuration options
- **Performance**: Benchmarks and optimization guidance in component docs

## Documentation Conventions

- **Purpose**: What the component does and why it exists
- **Technical Implementation**: Core algorithms and data structures
- **Configuration**: Key parameters and tuning options
- **Performance**: Speed, memory, and quality characteristics
- **Use Cases**: When and how to use the component

Each component is documented with both conceptual understanding and practical implementation details for effective usage and modification.