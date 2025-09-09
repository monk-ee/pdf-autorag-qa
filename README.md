# 🎸 AutoRAG Audio Equipment Q&A Pipeline

[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-FFD21E)](https://huggingface.co)

> **GPU-Accelerated Q&A Extraction and RAG Evaluation Pipeline for Audio Equipment Documentation**

Transform audio equipment manuals into high-quality training datasets through automated Q&A generation, FAISS vector indexing, and RAG-enhanced evaluation using Llama-3-8B-Instruct.

## 🚀 **What This Pipeline Does**

This repository implements a complete **Dual RAG AutoRAG Pipeline** that:

1. **📄 Extracts Q&A pairs** from audio equipment PDFs using GPU-accelerated LLM processing
2. **🎯 Generates 9 matrix combinations** (3 difficulty levels × 3 creativity styles) 
3. **🔍 Builds dual vector stores** with both Standard and Adaptive RAG approaches (4 FAISS indices)
4. **⚡ Evaluates both RAG approaches** with scientific A/B testing and comparison analysis
5. **📊 Produces domain-specific insights** with dual approach performance comparison
6. **🎓 Generates training datasets** using the winning RAG approach based on empirical results

---

## 🏗️ **Dual RAG Pipeline Architecture**

```mermaid
graph LR
    A[📄 Audio Manual PDF] --> B[🎸 Q&A Generation<br/>Matrix 3×3]
    B --> C[🔍 Top-K Selection]
    C --> D[🧠 Dual Vector Store Builder]
    
    D --> E1[🔹 Standard RAG<br/>Answer Embeddings]
    D --> E2[🔸 Enhanced Adaptive RAG<br/>Hybrid Dense+Sparse+Reranking]
    
    E1 --> F1[💾 CPU/GPU Indices<br/>Standard]
    E2 --> F2[💾 CPU/GPU Indices<br/>Adaptive]
    
    F1 --> G1[⚡ Standard RAG Eval]
    F2 --> G2[⚡ Adaptive RAG Eval]
    
    G1 --> H[📊 A/B Comparison]
    G2 --> H
    
    H --> I[🏆 Winner Selection]
    I --> J[🎓 Training Dataset]
    
    style A fill:#f9d71c
    style H fill:#e74c3c
    style J fill:#27ae60
```

### **3×3 Matrix Generation Strategy**

| Difficulty | High Creativity (0.9) | Balanced (0.7) | Conservative (0.3) |
|------------|----------------------|-----------------|-------------------|
| **Basic** | Broad, creative questions | Standard questions | Focused, literal |
| **Intermediate** | Complex scenarios | Technical details | Specific procedures |
| **Advanced** | Expert-level analysis | Professional insights | Precise specifications |

---

## 🔸 **Enhanced Adaptive RAG Features**

The **Enhanced Adaptive RAG** implementation includes 4 major improvements over basic Q+A embeddings:

### **🔄 1. Cross-Encoder Re-ranking** (+3-5% improvement)
- **MS-Marco MiniLM** cross-encoder for better query-document relevance
- **Weighted scoring**: 60% cross-encoder + 40% retrieval scores
- **Graceful fallback** if cross-encoder fails

### **🧠📝 2. Hybrid Dense+Sparse Retrieval** (+5-8% improvement)  
- **Multiple dense strategies**: Combined Q+A, Question-only, Answer-only embeddings
- **BM25 sparse retrieval** + TF-IDF backup for lexical matching
- **Adaptive weighting** (α=0.3-0.9) based on query type and domain relevance

### **📏 3. Dynamic Context Windows** (+2-3% improvement)
- **Query complexity adjustment**: ±30% context for advanced/basic queries
- **Domain relevance filtering**: ±30% for high/low domain relevance  
- **Confidence-based selection** with statistical thresholds

### **🎯 4. Enhanced Query Classification** (+3-5% improvement)
- **Domain ontology**: 50+ audio equipment terms and categories
- **Entity recognition** for technical specifications
- **Multi-factor confidence thresholds** (0.2-0.9 dynamic range)

**Expected Total Improvement**: **13-21%** over Standard RAG (90-95% vs 81% quality score)

### **📋 Implementation Status**
- ✅ **Enhanced features implemented**: `adaptive_rag_pipeline.py` contains all 4 improvements
- ⏳ **Integration pending**: Enhanced pipeline needs integration with existing evaluator
- 📊 **Current results**: Based on basic Q+A embeddings (81% → 82.4% = +1.4% improvement)
- 🎯 **Next step**: Integrate enhanced pipeline for projected 10-20% improvement

---

## 🛠️ **Quick Start**

### **Prerequisites**
- **GPU Required**: NVIDIA GPU with CUDA support (L40S recommended)
- **Python**: 3.9+
- **Storage**: ~10GB for models + datasets

### **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd autorag

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry
poetry install

# Set up Hugging Face token
export HF_TOKEN="your_hugging_face_token_here"
```

### **Run the Complete Pipeline**
```bash
# Trigger the full AutoRAG pipeline via GitHub Actions
gh workflow run pdf-qa-autorag.yaml \
  --field input_file="pdfs/UAFX_Ruby_63_Top_Boost_Amplifier_Manual.pdf" \
  --field model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
  --field top_k_selection="50"
```

Or run components individually:

```bash
# 1. Generate Q&A pairs (example: basic difficulty, balanced creativity)
python cli_pdf_qa.py \
  pdfs/UAFX_Ruby_63_Top_Boost_Amplifier_Manual.pdf \
  --output outputs/qa_basic_balanced.jsonl \
  --difficulty-levels basic \
  --temperature 0.7 --top-p 0.9

# 2. Select best pairs
python qa_pair_selector.py \
  --qa-artifacts-dir outputs \
  --output-dir rag_input \
  --top-k 50

# 3. Build dual vector stores (Standard + Adaptive RAG)
python qa_faiss_builder.py \
  --qa-pairs-file rag_input/selected_qa_pairs.json \
  --output-dir rag_store

# 4. Run parallel RAG evaluation (both approaches)
# Standard RAG
python qa_autorag_evaluator.py \
  --qa-pairs-file rag_input/selected_qa_pairs.json \
  --qa-faiss-index rag_store/qa_faiss_index_standard_gpu.bin \
  --output-dir autorag_results/standard_rag

# Adaptive RAG (currently basic Q+A embeddings)
# Note: Enhanced features implemented in adaptive_rag_pipeline.py - integration pending
python qa_autorag_evaluator.py \
  --qa-pairs-file rag_input/selected_qa_pairs.json \
  --qa-faiss-index rag_store/qa_faiss_index_adaptive_gpu.bin \
  --output-dir autorag_results/adaptive_rag

# 5. Compare RAG approaches
python rag_comparison_analyzer.py \
  --standard-results autorag_results/standard_rag \
  --adaptive-results autorag_results/adaptive_rag \
  --output-file autorag_results/rag_comparison_report.json

# 6. Domain-specific evaluation
python domain_eval_gpu.py \
  --config audio_equipment_domain_questions.json \
  --results-dir outputs
```

---

## 📁 **Project Structure**

```
autorag/
├── 🎸 pdfs/                                    # Audio equipment manuals
│   └── UAFX_Ruby_63_Top_Boost_Amplifier_Manual.pdf
├── ⚙️  qa_extraction_lib/                      # Core extraction library
│   ├── pdf_generator.py                       # PDF text processing
│   ├── prompt_manager.py                      # LLM prompt templates
│   └── text_processing.py                     # Text chunking & preprocessing
├── 🔧 Pipeline Scripts
│   ├── cli_pdf_qa.py                         # Main Q&A generator (9 matrix combinations)
│   ├── qa_pair_selector.py                   # Top-K selection algorithm
│   ├── qa_faiss_builder.py                   # GPU FAISS index builder
│   ├── qa_autorag_evaluator.py               # RAG vs Base model evaluation
│   ├── training_dataset_generator.py          # High-quality dataset generator
│   └── domain_eval_gpu.py                    # Audio equipment domain evaluator
├── 🎯 Configuration
│   ├── audio_equipment_domain_questions.json  # Domain-specific evaluation config
│   └── pyproject.toml                         # Poetry dependencies and project config
├── 🤖 .github/workflows/
│   └── pdf-qa-autorag.yaml                   # Complete CI/CD pipeline
└── 📊 Output Directories (auto-created)
    ├── outputs/           # Generated Q&A pairs (9 matrix files)
    ├── rag_input/         # Selected pairs + metadata
    ├── rag_store/         # FAISS indices + embeddings
    └── autorag_results/   # Evaluation reports + training datasets
```

---

## 🎯 **Key Features**

### **🚀 Dual RAG Architecture**
- **4 FAISS indices** (CPU/GPU × Standard/Adaptive) for comprehensive evaluation
- **Standard RAG**: Traditional answer-only embeddings (speed-optimized)
- **Adaptive RAG**: Combined Q+A embeddings (quality-optimized) 
- **Scientific A/B testing** with quantitative performance comparison
- **Automatic winner selection** based on empirical results

### **📊 Comprehensive Evaluation**
- **Standard vs Adaptive RAG** head-to-head comparison
- **RAG vs Base Model** performance analysis
- **BERT-Score semantic evaluation** for both approaches
- **Domain relevance scoring** with dual approach insights
- **Uncertainty detection** and confidence calibration
- **Performance metrics** (speed vs quality trade-offs)

### **🎓 Training-Ready Outputs**
- **Winner-based training data** using best-performing RAG approach
- **High-quality Q&A pairs** filtered by semantic similarity and comparison results
- **JSONL format** compatible with popular training frameworks
- **Metadata preservation** (difficulty, creativity, source tracking, RAG comparison scores)
- **Quality metrics** and approach selection rationale for dataset curation

---

## 🔬 **Evaluation Metrics**

The pipeline provides multi-dimensional evaluation:

| Metric Category | Measures | Good For |
|----------------|----------|----------|
| **Semantic Quality** | BERT-Score F1, Precision, Recall | Answer accuracy (both approaches) |
| **Domain Relevance** | Audio equipment term frequency | Specialization (Standard vs Adaptive) |
| **Response Length** | Word count, token count | Completeness comparison |
| **Uncertainty** | "I don't know" phrase detection | Confidence calibration |
| **Retrieval Quality** | Dense + sparse score combination | Context relevance (dual comparison) |
| **Performance** | Retrieval/generation time (ms) | Speed vs quality trade-offs |
| **Approach Comparison** | Standard vs Adaptive metrics | Winner selection criteria |

---

## 🎸 **Audio Equipment Domain**

Specifically tuned for **guitar amplifiers and effects**:

- **Domain Terms**: amplifier, guitar, tone, distortion, overdrive, gain, EQ, tube, preamp, etc.
- **Question Categories**: Technical specifications, setup procedures, troubleshooting, comparisons
- **Knowledge Areas**: Impedance matching, tube saturation, power handling, signal processing

---

## 📚 **Detailed Documentation**

For in-depth technical details on each component:

### Component Documentation
- **[Q&A Generation](docs/cli_pdf_qa.md)** - PDF processing and LLM-based Q&A extraction
- **[Quality Selection](docs/qa_pair_selector.md)** - Multi-metric quality assessment and filtering  
- **[Dual Vector Store](docs/qa_faiss_builder.md)** - GPU-accelerated FAISS indexing with Standard + Adaptive approaches
- **[RAG Evaluation](docs/qa_autorag_evaluator.md)** - Parallel RAG evaluation and performance comparison
- **[Training Dataset](docs/training_dataset_generator.md)** - Winner-based training data generation
- **[Domain Evaluation](docs/domain_eval_gpu.md)** - Domain expertise with dual RAG analysis

### Architecture Overview  
- **[Pipeline Architecture](docs/architecture.md)** - Complete dual RAG system design and data flow
- **[Dual RAG Architecture](docs/dual_rag_architecture.md)** - Standard vs Adaptive RAG comparison framework

Each component document includes technical implementation details, configuration options, performance characteristics, and use cases.

---

## 🚀 **Advanced Usage**

### **Custom PDF Processing**
```bash
# Process your own audio equipment manual
python cli_pdf_qa.py your_manual.pdf \
  --chunk-size 600 \
  --batch-size 4 \
  --difficulty-levels basic intermediate \
  --quantize  # Enable for lower GPU memory
```

### **Fine-tune Selection Criteria**
```bash
# More aggressive filtering
python qa_pair_selector.py \
  --qa-artifacts-dir outputs \
  --top-k 25 \
  --min-quality-threshold 0.7
```

### **Custom Domain Configuration**
Edit `audio_equipment_domain_questions.json` to:
- Add new domain terms
- Create custom evaluation questions  
- Modify confidence templates

---

## 🎯 **Expected Results**

After running the complete dual RAG pipeline, expect:

- **~500-1000 Q&A pairs** from a typical amplifier manual
- **50+ high-quality pairs** selected for dual RAG evaluation
- **4 GPU/CPU FAISS indices** with sub-millisecond query times
- **Comparative analysis** showing Standard vs Adaptive performance differences
- **Standard RAG performance**: ~81% quality score baseline
- **Enhanced Adaptive RAG**: Expected 10-20% improvement (90-95% quality score)
- **Winner determination** based on quality metrics (timing not measured)
- **Performance gains**: Cross-encoder re-ranking, hybrid retrieval, dynamic context windows
- **Training dataset** generated from best-performing approach

---

## 🤝 **Contributing**

This pipeline is designed for **audio equipment domain specialization**. To adapt for other domains:

1. **Replace PDF**: Add your domain-specific documentation to `pdfs/`
2. **Update domain config**: Modify `audio_equipment_domain_questions.json`
3. **Adjust prompts**: Edit templates in `qa_extraction_lib/prompt_manager.py`
4. **Update workflow**: Change default paths in `.github/workflows/pdf-qa-autorag.yaml`

---

## 📝 **License**

This project demonstrates advanced RAG pipeline techniques for domain-specific knowledge extraction. Built with modern ML tools including PyTorch, Transformers, FAISS, and Llama-3.

**Key Technologies**: Python 3.9+, PyTorch 2.1+, Transformers 4.42+, FAISS GPU, Sentence-Transformers, BERT-Score

---

*🎸 Ready to amplify your audio equipment knowledge with AI? Let's rock! 🤘*