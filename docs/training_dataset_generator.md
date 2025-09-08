# Training Dataset Generation (training_dataset_generator.py)

## Purpose
Converts RAG evaluation results into production-ready training datasets by filtering for highest-quality Q&A pairs using performance metrics.

## The Quality-Performance Connection
Not all Q&A pairs are equal for training. This component identifies pairs where RAG significantly outperforms base models - indicating high-value training examples.

## Quality Selection Algorithm

### Performance-Based Filtering
```python
high_quality_pairs = pairs.filter(
    bert_f1_improvement > 0.1,      # RAG significantly better
    domain_relevance > 0.6,         # Technical vocabulary present
    answer_length_ratio < 3.0,      # Reasonable response length
    uncertainty_indicators == False  # Confident responses
)
```

### Multi-Dimensional Quality Assessment
1. **RAG Improvement**: Pairs where retrieval context clearly helps
2. **Semantic Quality**: High BERT-Score ratings
3. **Domain Specificity**: Rich technical terminology
4. **Response Coherence**: Well-structured, complete answers

## Technical Implementation

### Quality Metrics Integration
```python
class TrainingDatasetGenerator:
    - load_evaluation_results()   # RAG performance metrics
    - calculate_quality_scores()  # Composite quality assessment
    - apply_quality_filters()     # Multi-threshold filtering
    - generate_training_format()  # Convert to standard formats
```

### Filtering Pipeline
1. **Load RAG Results**: Import evaluation metrics and responses
2. **Score Calculation**: Weighted combination of quality metrics
3. **Threshold Application**: Remove low-performing examples
4. **Format Conversion**: Export in training-ready formats

## Quality Criteria

### Primary Filters (Must Pass)
- **BERT-Score F1** > 0.7: Strong semantic quality
- **RAG Improvement** > 0.05: Context demonstrably helps
- **Answer Completeness**: Minimum 20 words, maximum 500 words
- **Error-Free**: No generation errors or malformed responses

### Secondary Scoring (Weighted)
- **Domain Relevance** (30%): Technical term density
- **Linguistic Quality** (25%): Grammar and clarity
- **Information Density** (25%): Concept coverage per word
- **Uniqueness** (20%): Non-redundant information

### Quality Score Formula
```python
quality_score = (
    0.30 * domain_relevance_score +
    0.25 * linguistic_quality_score +
    0.25 * information_density_score +
    0.20 * uniqueness_score
) * rag_improvement_bonus
```

## Output Formats

### Standard Training Format (JSONL)
```json
{
  "instruction": "What is the purpose of the tube preamp section?",
  "input": "",
  "output": "The tube preamp section provides initial signal amplification with characteristic harmonic distortion and compression that defines the amplifier's tonal character.",
  "quality_metrics": {
    "bert_f1": 0.847,
    "rag_improvement": 0.156,
    "domain_relevance": 0.723,
    "composite_score": 0.782
  },
  "metadata": {
    "source_pdf": "UAFX_Ruby_63_Top_Boost_Amplifier_Manual.pdf",
    "difficulty": "intermediate",
    "creativity": "balanced",
    "context_pairs_used": 3
  }
}
```

### Training Framework Compatibility
- **Alpaca Format**: Instruction-input-output structure
- **Conversation Format**: Multi-turn dialogue adaptation
- **Custom Schema**: Flexible field mapping

## Dataset Characteristics

### Quality Distribution
- **High Quality** (Score ≥ 0.8): Top 10-15% of generated pairs
- **Medium Quality** (Score 0.6-0.8): Training data backbone (60-70%)
- **Low Quality** (Score < 0.6): Filtered out

### Typical Dataset Composition
From 50 selected Q&A pairs → ~25-35 training examples:
- **Factual Questions**: 40-50% (high success rate)
- **Procedural Questions**: 30-40% (moderate complexity)
- **Analytical Questions**: 10-20% (challenging but valuable)

### Difficulty Balance
- **Basic**: 30% (foundation knowledge)
- **Intermediate**: 50% (practical application)
- **Advanced**: 20% (expert-level concepts)

## Advanced Features

### Adaptive Quality Thresholds
```python
def calculate_dynamic_thresholds(evaluation_results):
    # Adjust thresholds based on overall performance distribution
    bert_threshold = percentile(evaluation_results.bert_f1, 75)
    improvement_threshold = median(evaluation_results.rag_improvement)
    return bert_threshold, improvement_threshold
```

### Contamination Detection
- **Near-Duplicate Removal**: Semantic clustering to avoid repetition
- **Test Set Leakage**: Ensure no overlap with evaluation questions
- **Domain Drift**: Monitor for off-topic content

## Configuration Options

### Quality Control
- `--min-bert-f1`: Minimum semantic quality threshold
- `--min-improvement`: Required RAG performance gain
- `--max-length-ratio`: Answer length reasonableness
- `--domain-weight`: Importance of technical vocabulary

### Dataset Size Management
- `--target-size`: Desired number of training examples
- `--quality-vs-quantity`: Balance between selectivity and volume
- `--diversity-sampling`: Ensure broad topic coverage

## Performance Validation

### Quality Assurance Process
1. **Automated Metrics**: Verify all quality thresholds met
2. **Sample Review**: Manual inspection of random subset
3. **Consistency Checks**: Validate instruction-output alignment
4. **Domain Coverage**: Ensure representative topic distribution

### Expected Outcomes
- **Retention Rate**: 50-70% of input pairs pass quality filters
- **Quality Improvement**: 2-3x better performance than random selection
- **Training Effectiveness**: Specialized models show measurable domain improvement

## Use Cases
- Fine-tuning datasets for domain-specific models
- Quality control for large-scale training data
- Research into optimal training example characteristics
- Benchmarking RAG system effectiveness through training outcomes