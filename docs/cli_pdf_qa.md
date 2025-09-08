# Q&A Generation Engine (cli_pdf_qa.py)

## Purpose
Extracts Q&A pairs from PDF documents using GPU-accelerated language models with configurable difficulty and creativity parameters.

## How It Works

### Text Processing Pipeline
1. **PDF Extraction**: PyMuPDF extracts raw text while preserving structure
2. **Chunking**: Splits text into overlapping windows (default: 800 words, 100-word overlap)
3. **Chunk Scoring**: Filters chunks based on information density and readability

### Q&A Generation Strategy
Uses prompt templates to generate questions at different complexity levels:

- **Basic**: Direct factual questions about explicit information
- **Intermediate**: Analytical questions requiring understanding of relationships  
- **Advanced**: Synthesis questions combining multiple concepts

### Creativity Control via Sampling
- **Conservative (T=0.3)**: Focused, literal questions with low variation
- **Balanced (T=0.7)**: Standard technical questions with moderate creativity
- **High Creativity (T=0.9)**: Exploratory questions with diverse phrasings

### GPU Optimization
- **Model Loading**: Llama-3-8B-Instruct with FP16 precision
- **Batch Processing**: Processes multiple chunks simultaneously
- **Memory Management**: Optional 8-bit quantization for 24GB+ models

## Technical Implementation

### Core Components
```python
class PDFQAExtractor:
    - load_model()      # Initialize Llama-3 with GPU acceleration
    - extract_text()    # PDF → structured text chunks
    - generate_qa()     # Chunk → Q&A pairs via LLM
    - filter_quality()  # Remove low-quality outputs
```

### Prompt Engineering
- Domain-specific vocabulary injection
- Context-aware question generation
- Answer grounding to source text
- Hallucination prevention through strict prompting

## Configuration

### Key Parameters
- `--difficulty-levels`: basic, intermediate, advanced
- `--temperature`: Sampling randomness (0.1-1.0)
- `--chunk-size`: Text window size in words
- `--batch-size`: GPU memory vs speed tradeoff
- `--quantize`: Enable 8-bit model compression

### Output Format
```json
{
  "instruction": "What is the purpose of the preamp gain control?",
  "input": "",
  "output": "The preamp gain control adjusts input sensitivity...",
  "source_chunk": "Original PDF text segment",
  "difficulty": "basic",
  "chunk_id": "chunk_042",
  "quality_score": 0.87
}
```

## Performance Characteristics

### Typical Throughput
- **L40S GPU**: ~50 Q&A pairs/minute with quantization
- **Memory Usage**: 12-16GB VRAM (8-bit), 20-24GB (FP16)
- **Quality vs Speed**: Higher temperatures = more creativity but slower generation

### Expected Outputs
- **Per Difficulty Level**: ~50-100 pairs from typical manual
- **Total (3×3 Matrix)**: 500-1000 pairs across all combinations
- **Quality Distribution**: 70-80% meet basic quality thresholds

## Use Cases
- Technical documentation → training data
- Domain-specific Q&A dataset creation
- Knowledge extraction from manuals
- Multi-level educational content generation