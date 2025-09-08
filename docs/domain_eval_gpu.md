# Domain Specificity Evaluation (domain_eval_gpu.py)

## Purpose
Evaluates how well the pipeline generates domain-specific knowledge by testing base vs RAG models on audio equipment questions with configurable domain criteria.

## The Domain Expertise Problem
Generic language models know a little about everything but lack deep domain expertise. This component measures whether our RAG approach successfully creates domain-specialized AI behavior.

## Evaluation Design

### Domain Configuration System
Uses `audio_equipment_domain_questions.json` to define:
- **Domain Vocabulary**: Technical terms (amplifier, gain, impedance, etc.)
- **Test Questions**: In-domain vs out-of-domain evaluation sets
- **Expected Behaviors**: Confidence patterns and uncertainty handling

### Test Question Categories
```json
{
  "in_domain_factual": "What is impedance matching in audio equipment?",
  "in_domain_procedural": "How do you troubleshoot amplifier noise?",
  "out_domain_general": "How do you change car engine oil?",
  "edge_case_ambiguous": "Which amplifier sounds best?"
}
```

## Technical Architecture

### GPU-Accelerated Evaluation
- **Model Loading**: Llama-3-8B with optional 8-bit quantization
- **Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) search
- **Batch Processing**: Parallel question evaluation

### Confidence-Gated Context Templates
```python
def format_context_by_confidence(question, retrieved_pairs):
    confidence = calculate_retrieval_confidence(retrieved_pairs)
    
    if confidence < 0.4:
        # Low confidence - encourage uncertainty
        return "If you cannot find sufficient information, say so."
    elif confidence < 0.7:
        # Medium confidence - cautious responses
        return "Based on available information..."
    else:
        # High confidence - comprehensive answers
        return "Use the reference information to provide detailed answers."
```

## Domain-Specific Metrics

### Technical Term Analysis
- **Domain Relevance Score**: Frequency of specialized vocabulary
- **Term Density**: Technical terms per response word
- **Concept Coverage**: Breadth of domain concepts mentioned

### Response Quality Assessment
- **Uncertainty Handling**: Appropriate "I don't know" responses for out-of-domain
- **Confidence Calibration**: Matching certainty to retrieval quality
- **Hallucination Prevention**: Reduced false claims on unfamiliar topics

### Comparative Analysis
```python
domain_improvement = {
    'factual_accuracy': rag_accuracy - base_accuracy,
    'technical_terminology': rag_term_usage - base_term_usage,
    'appropriate_uncertainty': base_uncertainty - rag_uncertainty,
    'answer_depth': rag_detail_level - base_detail_level
}
```

## Core Components

### Evaluation Pipeline
```python
class DomainEvaluatorGPU:
    - load_domain_config()      # Audio equipment evaluation setup
    - build_test_questions()    # Mix of in/out domain questions  
    - evaluate_responses()      # Base vs RAG comparison
    - analyze_domain_metrics()  # Domain-specific analysis
```

### Context Template System
- **Comparison Template**: For A vs B technical questions
- **Technical Template**: For specification-heavy questions
- **General Template**: Default with confidence gating

## Evaluation Results

### Domain Expertise Indicators
- **In-Domain Performance**: Should significantly improve with RAG
- **Out-Domain Behavior**: Should show appropriate uncertainty
- **Technical Vocabulary**: Increased usage of specialized terms
- **Answer Depth**: More detailed, nuanced responses

### Expected Performance Patterns
```json
{
  "in_domain_questions": {
    "base_domain_relevance": 0.3-0.5,
    "rag_domain_relevance": 0.6-0.8,
    "improvement": "60-100%"
  },
  "out_domain_questions": {
    "base_uncertainty_rate": 0.1-0.3,
    "rag_uncertainty_rate": 0.4-0.7,
    "improvement": "Increased appropriate uncertainty"
  }
}
```

## Advanced Features

### Configurable Domain Adaptation
- **Custom Vocabularies**: Easy domain switching via JSON config
- **Template Customization**: Domain-specific prompt patterns
- **Metric Weighting**: Adjust importance of different quality factors

### Multi-Domain Evaluation
- **Cross-Domain Testing**: Audio equipment model on medical questions
- **Domain Transfer**: Measure knowledge boundary sharpness
- **Contamination Detection**: Identify unintended domain bleeding

## Configuration

### Domain Setup
- `--config`: Domain configuration JSON file
- `--model`: Base language model for evaluation
- `--quantize`: Enable memory-efficient model loading

### Evaluation Parameters
- `--max-questions`: Number of test questions per category
- `--no-bert-score`: Disable semantic similarity evaluation
- `--results-dir`: Directory containing generated Q&A pairs

## Performance Analysis

### GPU Utilization
- **Model Inference**: Llama-3-8B evaluation queries
- **Embedding Generation**: Sentence transformer for retrieval
- **FAISS Search**: GPU-accelerated similarity search

### Evaluation Speed
- **Typical Runtime**: 10-15 minutes for full evaluation
- **Memory Requirements**: 16-24GB GPU VRAM
- **Scalability**: Linear with question count

## Output Reports

### Domain Analysis Summary
```json
{
  "domain_effectiveness": {
    "avg_domain_relevance_improvement": 0.342,
    "technical_term_usage_increase": 67.8,
    "appropriate_uncertainty_improvement": 0.234
  },
  "by_question_category": {
    "factual": {"improvement": 0.187, "confidence": "high"},
    "procedural": {"improvement": 0.156, "confidence": "medium"},
    "analytical": {"improvement": 0.092, "confidence": "low"}
  }
}
```

### Quality Validation
- **Manual Review Support**: Detailed response logging
- **Metric Correlation**: Compare automated scores with human judgment
- **Failure Analysis**: Identify common error patterns

## Use Cases
- Domain adaptation effectiveness measurement
- RAG system specialization validation
- Training data quality assessment via domain performance
- Research into domain-specific AI behavior patterns
- Multi-domain model boundary analysis