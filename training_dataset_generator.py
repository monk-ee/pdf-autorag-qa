#!/usr/bin/env python3
"""
Training Dataset Generator for AutoRAG Pipeline

Creates final high-quality training dataset from RAG evaluation results.
Filters pairs based on quality thresholds and generates training formats.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_evaluation_results(results_file: Path) -> List[Dict[str, Any]]:
    """
    Load RAG evaluation results from JSON file.
    
    Args:
        results_file: Path to evaluation results JSON
        
    Returns:
        List of evaluation results
    """
    print(f'📥 Loading evaluation results: {results_file}')
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both direct list format and report format from qa_autorag_evaluator
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and 'detailed_results' in data:
        results = data['detailed_results']
        print(f'📊 Loaded report format with evaluation summary')
    else:
        raise ValueError(f'Unexpected data format in {results_file}. Expected list or dict with detailed_results.')
    
    print(f'📊 Loaded {len(results)} evaluation results')
    return results


def filter_high_quality_pairs(evaluation_results: List[Dict[str, Any]], 
                             quality_thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Filter high-quality pairs based on evaluation metrics.
    
    Args:
        evaluation_results: List of evaluation results
        quality_thresholds: Dictionary of quality thresholds
        
    Returns:
        List of filtered high-quality pairs
    """
    high_quality_pairs = []
    
    min_similarity = quality_thresholds.get('minimum_similarity', 0.3)
    min_length_ratio = quality_thresholds.get('minimum_length_ratio', 0.5)
    max_length_ratio = quality_thresholds.get('maximum_length_ratio', 2.0)
    min_answer_length = quality_thresholds.get('minimum_answer_length', 20)
    min_completeness = quality_thresholds.get('minimum_completeness', 0.2)
    
    for result in evaluation_results:
        # Skip failed evaluations
        if 'error' in result:
            continue
            
        metrics = result.get('metrics', {})
        semantic_similarity = metrics.get('semantic_similarity', 0.0)
        content_overlap = metrics.get('content_overlap', 0.0) 
        length_ratio = metrics.get('length_ratio', 0.0)
        quality_score = metrics.get('quality_score', 0.0)
        rag_answer = result.get('rag_answer', '')
        
        # Apply quality filters using quality_score as primary metric
        # Use semantic_similarity as backup if quality_score is 0
        primary_similarity = quality_score if quality_score > 0 else semantic_similarity
        
        if (primary_similarity >= min_similarity and
            min_length_ratio <= length_ratio <= max_length_ratio and
            len(rag_answer.strip()) >= min_answer_length and
            content_overlap >= min_completeness):
            
            high_quality_pairs.append({
                'question': result['question'],
                'answer': result['original_answer'],  # Use original answer
                'rag_answer': result['rag_answer'],   # Include RAG verification
                'quality_score': primary_similarity,
                'semantic_similarity': semantic_similarity,
                'content_overlap': content_overlap,
                'length_ratio': length_ratio,
                'source_matrix': result.get('matrix_combination', 'unknown'),
                'pair_id': result.get('pair_id', -1),
                'verified': True,
                'evaluation_metrics': metrics
            })
    
    print(f'✅ Selected {len(high_quality_pairs)} high-quality pairs from {len(evaluation_results)} total')
    return high_quality_pairs


def generate_training_formats(high_quality_pairs: List[Dict[str, Any]], 
                            output_dir: Path) -> Dict[str, Path]:
    """
    Generate training dataset in multiple formats.
    
    Args:
        high_quality_pairs: List of high-quality Q&A pairs
        output_dir: Output directory
        
    Returns:
        Dictionary mapping format names to output file paths
    """
    output_files = {}
    
    # JSONL format (most common for fine-tuning)
    jsonl_path = output_dir / 'high_quality_pairs.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for pair in high_quality_pairs:
            training_sample = {
                'question': pair['question'],
                'answer': pair['answer']
            }
            f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    output_files['jsonl'] = jsonl_path
    
    # Detailed JSONL with metadata (for analysis)
    detailed_jsonl_path = output_dir / 'high_quality_pairs_detailed.jsonl'
    with open(detailed_jsonl_path, 'w', encoding='utf-8') as f:
        for pair in high_quality_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    output_files['detailed_jsonl'] = detailed_jsonl_path
    
    # JSON format (complete dataset)
    json_path = output_dir / 'high_quality_pairs.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(high_quality_pairs, f, indent=2, ensure_ascii=False)
    output_files['json'] = json_path
    
    # Hugging Face datasets format
    hf_dataset_path = output_dir / 'hf_dataset.json'
    hf_format = []
    for pair in high_quality_pairs:
        hf_format.append({
            'instruction': pair['question'],
            'input': '',  # Empty input for Q&A format
            'output': pair['answer']
        })
    
    with open(hf_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(hf_format, f, indent=2, ensure_ascii=False)
    output_files['huggingface'] = hf_dataset_path
    
    # ChatML format (for chat models)
    chatml_path = output_dir / 'chatml_format.jsonl'
    with open(chatml_path, 'w', encoding='utf-8') as f:
        for pair in high_quality_pairs:
            chatml_sample = {
                'messages': [
                    {'role': 'user', 'content': pair['question']},
                    {'role': 'assistant', 'content': pair['answer']}
                ]
            }
            f.write(json.dumps(chatml_sample, ensure_ascii=False) + '\n')
    output_files['chatml'] = chatml_path
    
    print(f'💾 Generated training datasets in {len(output_files)} formats')
    return output_files


def analyze_matrix_performance(high_quality_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze performance by matrix combination.
    
    Args:
        high_quality_pairs: List of high-quality pairs
        
    Returns:
        Performance analysis by matrix combination
    """
    matrix_performance = {}
    
    for pair in high_quality_pairs:
        matrix = pair['source_matrix']
        if matrix not in matrix_performance:
            matrix_performance[matrix] = {
                'count': 0,
                'quality_scores': [],
                'avg_quality': 0.0,
                'avg_length_ratio': 0.0,
                'avg_completeness': 0.0
            }
        
        stats = matrix_performance[matrix]
        stats['count'] += 1
        stats['quality_scores'].append(pair['quality_score'])
    
    # Calculate averages
    for matrix, stats in matrix_performance.items():
        if stats['count'] > 0:
            stats['avg_quality'] = sum(stats['quality_scores']) / stats['count']
            
            # Calculate other averages from pairs
            matrix_pairs = [p for p in high_quality_pairs if p['source_matrix'] == matrix]
            if matrix_pairs:
                stats['avg_length_ratio'] = sum(p['length_ratio'] for p in matrix_pairs) / len(matrix_pairs)
                stats['avg_completeness'] = sum(p['content_overlap'] for p in matrix_pairs) / len(matrix_pairs)
    
    # Sort by average quality
    sorted_performance = dict(sorted(matrix_performance.items(), 
                                   key=lambda x: x[1]['avg_quality'], 
                                   reverse=True))
    
    return sorted_performance


def generate_final_report(high_quality_pairs: List[Dict[str, Any]],
                         evaluation_results: List[Dict[str, Any]],
                         quality_thresholds: Dict[str, float],
                         matrix_performance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive final evaluation report.
    
    Args:
        high_quality_pairs: Filtered high-quality pairs
        evaluation_results: Original evaluation results
        quality_thresholds: Quality thresholds used
        matrix_performance: Matrix performance analysis
        
    Returns:
        Final report dictionary
    """
    # Get top 3 matrix combinations
    top_combinations = list(matrix_performance.items())[:3]
    
    report = {
        'pipeline_summary': {
            'total_matrix_combinations': len(set(r.get('matrix_combination', 'unknown') 
                                               for r in evaluation_results)),
            'total_evaluated_pairs': len(evaluation_results),
            'successful_evaluations': len([r for r in evaluation_results if 'error' not in r]),
            'high_quality_pairs': len(high_quality_pairs),
            'quality_retention_rate': len(high_quality_pairs) / len(evaluation_results) if evaluation_results else 0
        },
        'quality_thresholds': quality_thresholds,
        'matrix_performance': matrix_performance,
        'recommendations': {
            'best_matrix_combinations': [
                {
                    'combination': combo,
                    'avg_quality': stats['avg_quality'],
                    'pair_count': stats['count'],
                    'avg_length_ratio': stats['avg_length_ratio'],
                    'avg_completeness': stats['avg_completeness']
                }
                for combo, stats in top_combinations
            ],
            'suggested_improvements': []
        },
        'dataset_statistics': {
            'total_training_pairs': len(high_quality_pairs),
            'avg_question_length': sum(len(p['question'].split()) for p in high_quality_pairs) / len(high_quality_pairs) if high_quality_pairs else 0,
            'avg_answer_length': sum(len(p['answer'].split()) for p in high_quality_pairs) / len(high_quality_pairs) if high_quality_pairs else 0,
            'avg_quality_score': sum(p['quality_score'] for p in high_quality_pairs) / len(high_quality_pairs) if high_quality_pairs else 0
        }
    }
    
    # Add improvement suggestions
    if len(high_quality_pairs) < len(evaluation_results) * 0.5:
        report['recommendations']['suggested_improvements'].append(
            "Consider lowering quality thresholds to increase dataset size"
        )
    
    if report['dataset_statistics']['avg_answer_length'] < 15:
        report['recommendations']['suggested_improvements'].append(
            "Consider increasing temperature or max_tokens for longer answers"
        )
    
    best_combo = top_combinations[0][0] if top_combinations else 'unknown'
    if 'conservative' in best_combo:
        report['recommendations']['suggested_improvements'].append(
            "Conservative settings performed best - consider using lower temperature for production"
        )
    elif 'high_creativity' in best_combo:
        report['recommendations']['suggested_improvements'].append(
            "High creativity settings performed best - consider using higher temperature for production"
        )
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate high-quality training dataset from RAG evaluation')
    parser.add_argument('--evaluation-results', type=Path, default='autorag_results/evaluation_results.json',
                       help='JSON file with RAG evaluation results')
    parser.add_argument('--output-dir', type=Path, default='autorag_results',
                       help='Output directory for training datasets')
    parser.add_argument('--min-similarity', type=float, default=0.3,
                       help='Minimum word overlap similarity threshold')
    parser.add_argument('--min-length-ratio', type=float, default=0.5,
                       help='Minimum length ratio threshold')
    parser.add_argument('--max-length-ratio', type=float, default=2.0,
                       help='Maximum length ratio threshold')
    parser.add_argument('--min-answer-length', type=int, default=20,
                       help='Minimum answer length (characters)')
    parser.add_argument('--min-completeness', type=float, default=0.2,
                       help='Minimum completeness score')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not args.evaluation_results.exists():
        print(f'❌ Evaluation results file not found: {args.evaluation_results}')
        return 1
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    try:
        # Load evaluation results
        evaluation_results = load_evaluation_results(args.evaluation_results)
        
        # Define quality thresholds
        quality_thresholds = {
            'minimum_similarity': args.min_similarity,
            'minimum_length_ratio': args.min_length_ratio,
            'maximum_length_ratio': args.max_length_ratio,
            'minimum_answer_length': args.min_answer_length,
            'minimum_completeness': args.min_completeness
        }
        
        # Filter high-quality pairs
        if args.verbose:
            print('🔍 Filtering high-quality pairs...')
        
        high_quality_pairs = filter_high_quality_pairs(evaluation_results, quality_thresholds)
        
        if not high_quality_pairs:
            print('❌ No pairs passed quality filtering. Consider lowering thresholds.')
            return 1
        
        # Generate training datasets
        if args.verbose:
            print('📝 Generating training datasets...')
        
        output_files = generate_training_formats(high_quality_pairs, args.output_dir)
        
        # Analyze matrix performance
        if args.verbose:
            print('📊 Analyzing matrix performance...')
        
        matrix_performance = analyze_matrix_performance(high_quality_pairs)
        
        # Generate final report
        if args.verbose:
            print('📋 Generating final report...')
        
        final_report = generate_final_report(
            high_quality_pairs, evaluation_results, 
            quality_thresholds, matrix_performance
        )
        
        # Save final report
        report_path = args.output_dir / 'final_evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print('🎊 Training dataset generation completed!')
        print(f'   📚 Training pairs: {len(high_quality_pairs)}')
        print(f'   📄 Generated formats: {", ".join(output_files.keys())}')
        print(f'   📈 Quality retention: {final_report["pipeline_summary"]["quality_retention_rate"]:.1%}')
        
        if matrix_performance:
            best_combo = list(matrix_performance.keys())[0]
            best_score = matrix_performance[best_combo]['avg_quality']
            print(f'   🏆 Best matrix combination: {best_combo} (quality: {best_score:.3f})')
        
        if args.verbose:
            print(f'   📋 Final report: {report_path}')
            for format_name, file_path in output_files.items():
                print(f'   📄 {format_name}: {file_path}')
        
        return 0
        
    except Exception as e:
        print(f'❌ Error generating training dataset: {e}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())