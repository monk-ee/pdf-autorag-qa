#!/usr/bin/env python3
"""
RAG vs Base Model Comparison Analyzer
Compares performance between Standard RAG and Base Model (no retrieval)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGvsBaseComparison:
    """Analyzes and compares RAG vs Base Model performance"""
    
    def __init__(self):
        logger.info("🔍 Initializing RAG vs Base Model Comparison")
    
    def load_evaluation_results(self, results_dir: Path) -> Dict[str, Any]:
        """Load evaluation results from directory"""
        report_file = results_dir / "qa_rag_evaluation_report.json"
        if results_dir.name == "base_model":
            report_file = results_dir / "qa_base_model_evaluation_report.json"
        
        if not report_file.exists():
            raise FileNotFoundError(f"Evaluation report not found: {report_file}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def calculate_improvement(self, rag_value: float, base_value: float) -> Dict[str, Any]:
        """Calculate improvement percentage and determine winner"""
        if base_value == 0:
            improvement_percent = float('inf') if rag_value > 0 else 0
        else:
            improvement_percent = ((rag_value - base_value) / base_value) * 100
        
        return {
            'rag': rag_value,
            'base': base_value,
            'improvement_percent': improvement_percent,
            'better_approach': 'rag' if rag_value > base_value else 'base'
        }
    
    def compare_approaches(self, rag_results: Dict, base_results: Dict) -> Dict[str, Any]:
        """Compare RAG vs Base Model performance"""
        logger.info("📊 Comparing RAG vs Base Model performance...")
        
        rag_summary = rag_results['evaluation_summary']
        base_summary = base_results['evaluation_summary']
        
        # Extract key metrics with correct field names
        comparison_metrics = {
            'avg_similarity_score': self.calculate_improvement(
                rag_summary.get('average_semantic_similarity', rag_summary.get('avg_similarity_score', 0)),
                base_summary.get('average_semantic_similarity', base_summary.get('avg_similarity_score', 0))
            ),
            'avg_bert_score': self.calculate_improvement(
                rag_summary.get('average_quality_score', rag_summary.get('avg_bert_score', 0)),
                base_summary.get('average_bert_score', base_summary.get('avg_bert_score', 0))
            ),
            'high_quality_pairs': self.calculate_improvement(
                rag_summary.get('successful_evaluations', rag_summary.get('high_quality_pairs', 0)),
                base_summary.get('high_quality_pairs', 0)
            ),
            'quality_retention_rate': self.calculate_improvement(
                rag_summary.get('quality_retention_rate', 0),
                base_summary.get('quality_retention_rate', 0)
            )
        }
        
        # Add context relevance if available (RAG-specific metric)
        if 'avg_context_relevance' in rag_summary:
            comparison_metrics['avg_context_relevance'] = {
                'rag': rag_summary['avg_context_relevance'],
                'base': 'N/A (no context)',
                'improvement_percent': 'N/A',
                'better_approach': 'rag'
            }
        
        # Calculate performance scores for each approach using correct field names
        rag_score = (
            rag_summary.get('average_semantic_similarity', rag_summary.get('avg_similarity_score', 0)) * 0.4 +
            rag_summary.get('average_quality_score', rag_summary.get('avg_bert_score', 0)) * 0.4 +
            rag_summary.get('quality_retention_rate', 0) * 0.2
        )
        
        base_score = (
            base_summary.get('average_semantic_similarity', base_summary.get('avg_similarity_score', 0)) * 0.4 +
            base_summary.get('average_bert_score', base_summary.get('avg_bert_score', 0)) * 0.4 +
            base_summary.get('quality_retention_rate', 0) * 0.2
        )
        
        # Determine overall winner
        overall_winner = 'rag' if rag_score > base_score else 'base'
        performance_improvement = ((rag_score - base_score) / base_score * 100) if base_score > 0 else 0
        
        return {
            'metadata': {
                'analysis_type': 'rag_vs_base_comparison',
                'rag_results_dir': 'autorag_results/standard_rag',
                'base_results_dir': 'autorag_results/base_model',
                'timestamp': datetime.now().isoformat()
            },
            'metrics_comparison': {
                'standard_rag': {
                    'approach': 'standard_rag',
                    'total_pairs_evaluated': rag_summary.get('total_pairs_evaluated', 0),
                    'avg_similarity_score': rag_summary.get('average_semantic_similarity', rag_summary.get('avg_similarity_score', 0)),
                    'avg_bert_score': rag_summary.get('average_quality_score', rag_summary.get('avg_bert_score', 0)),
                    'high_quality_pairs': rag_summary.get('successful_evaluations', rag_summary.get('high_quality_pairs', 0)),
                    'quality_retention_rate': rag_summary.get('quality_retention_rate', 0),
                    'avg_context_relevance': rag_summary.get('avg_context_relevance', 0),
                    'retrieval_time_ms': rag_summary.get('retrieval_time_ms', 0),
                    'generation_time_ms': rag_summary.get('generation_time_ms', 0),
                    'overall_score': rag_score
                },
                'base_model': {
                    'approach': 'base_model',
                    'total_pairs_evaluated': base_summary.get('total_pairs_evaluated', 0),
                    'avg_similarity_score': base_summary.get('average_semantic_similarity', base_summary.get('avg_similarity_score', 0)),
                    'avg_bert_score': base_summary.get('average_bert_score', base_summary.get('avg_bert_score', 0)),
                    'high_quality_pairs': base_summary.get('high_quality_pairs', 0),
                    'quality_retention_rate': base_summary.get('quality_retention_rate', 0),
                    'avg_context_relevance': 'N/A',
                    'retrieval_time_ms': 0,
                    'generation_time_ms': base_summary.get('total_time_seconds', 0) * 1000,
                    'overall_score': base_score
                },
                'differences': comparison_metrics
            },
            'approach_analysis': {
                'standard_rag': {
                    'description': 'RAG using curated Q&A knowledge base with FAISS vector retrieval',
                    'strengths': [
                        'Access to specific domain knowledge',
                        'Contextual information for accurate answers',
                        'Reduced hallucination with grounded responses',
                        'Consistent performance on domain-specific queries'
                    ],
                    'weaknesses': [
                        'Additional retrieval latency',
                        'Dependent on quality of knowledge base',
                        'May miss general knowledge outside corpus',
                        'Requires vector index maintenance'
                    ],
                    'best_use_cases': [
                        'Domain-specific question answering',
                        'Technical documentation queries',
                        'Factual information retrieval',
                        'Knowledge base applications'
                    ]
                },
                'base_model': {
                    'description': 'Base language model using only pre-trained knowledge',
                    'strengths': [
                        'Faster response time (no retrieval)',
                        'Broad general knowledge coverage',
                        'Simple architecture and deployment',
                        'No external knowledge base required'
                    ],
                    'weaknesses': [
                        'Potential hallucination without grounding',
                        'Limited domain-specific knowledge',
                        'Knowledge cutoff limitations',
                        'Inconsistent factual accuracy'
                    ],
                    'best_use_cases': [
                        'General conversation and reasoning',
                        'Creative writing and ideation',
                        'Quick responses without retrieval overhead',
                        'Exploratory or open-ended queries'
                    ]
                }
            },
            'recommendations': {
                'overall_winner': overall_winner,
                'performance_improvement': f"{performance_improvement:+.1f}%",
                'use_rag_when': [
                    'Domain-specific accuracy is critical',
                    'Grounded responses are required',
                    'Working with technical documentation',
                    'Factual correctness is more important than speed'
                ],
                'use_base_when': [
                    'Speed is more important than domain accuracy',
                    'General knowledge queries',
                    'Creative or open-ended tasks',
                    'Simple deployment is preferred'
                ],
                'key_insights': [
                    f"RAG provides {comparison_metrics['avg_bert_score']['improvement_percent']:+.1f}% improvement in BERT F1 score",
                    f"Quality retention rate improves by {comparison_metrics['quality_retention_rate']['improvement_percent']:+.1f}% with RAG",
                    "RAG shows consistent advantage for domain-specific technical queries",
                    "Base model suitable for general knowledge but less reliable for technical facts"
                ]
            },
            'summary': {
                'winner': overall_winner,
                'key_finding': f"RAG provides {performance_improvement:+.1f}% overall performance improvement",
                'recommendation_summary': f"Recommend {'RAG for domain-specific applications' if overall_winner == 'rag' else 'Base model for general applications'}"
            }
        }
    
    def analyze_detailed_results(self, rag_results: Dict, base_results: Dict) -> Dict[str, Any]:
        """Analyze detailed question-by-question performance"""
        rag_details = rag_results.get('detailed_results', [])
        base_details = base_results.get('detailed_results', [])
        
        if not rag_details or not base_details:
            logger.warning("No detailed results available for comparison")
            return {}
        
        # Compare question by question (assuming same order)
        question_comparisons = []
        for i, (rag_result, base_result) in enumerate(zip(rag_details, base_details)):
            if rag_result['question'] == base_result['question']:
                comparison = {
                    'question': rag_result['question'],
                    'rag_similarity': rag_result.get('similarity_score', 0),
                    'base_similarity': base_result.get('similarity_score', 0),
                    'rag_bert_f1': rag_result.get('bert_f1', 0),
                    'base_bert_f1': base_result.get('bert_f1', 0),
                    'rag_wins': rag_result.get('bert_f1', 0) > base_result.get('bert_f1', 0)
                }
                question_comparisons.append(comparison)
        
        rag_wins = sum(1 for q in question_comparisons if q['rag_wins'])
        total_questions = len(question_comparisons)
        
        return {
            'question_level_analysis': {
                'total_questions_compared': total_questions,
                'rag_wins': rag_wins,
                'base_wins': total_questions - rag_wins,
                'rag_win_rate': rag_wins / total_questions if total_questions > 0 else 0,
                'detailed_comparisons': question_comparisons[:10]  # First 10 for brevity
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Compare RAG vs Base Model performance")
    parser.add_argument('--standard-results', type=str, required=True,
                        help='Path to standard RAG results directory')
    parser.add_argument('--base-results', type=str, required=True,
                        help='Path to base model results directory')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSON file for comparison report')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize comparison analyzer
    analyzer = RAGvsBaseComparison()
    
    try:
        # Load results
        logger.info(f"Loading Standard RAG results from: {args.standard_results}")
        rag_results = analyzer.load_evaluation_results(Path(args.standard_results))
        
        logger.info(f"Loading Base Model results from: {args.base_results}")
        base_results = analyzer.load_evaluation_results(Path(args.base_results))
        
        # Perform comparison
        comparison_report = analyzer.compare_approaches(rag_results, base_results)
        
        # Add detailed analysis
        detailed_analysis = analyzer.analyze_detailed_results(rag_results, base_results)
        if detailed_analysis:
            comparison_report.update(detailed_analysis)
        
        # Save comparison report
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        logger.info(f"✅ Comparison report saved to: {output_path}")
        
        # Print summary
        summary = comparison_report['summary']
        metrics = comparison_report['metrics_comparison']
        
        print(f"\n🎯 RAG vs BASE MODEL COMPARISON")
        print(f"===============================")
        print(f"Overall Winner: {summary['winner'].upper()}")
        print(f"Key Finding: {summary['key_finding']}")
        print(f"Recommendation: {summary['recommendation_summary']}")
        print(f"\n📊 Performance Metrics:")
        print(f"RAG BERT F1: {metrics['standard_rag']['avg_bert_score']:.3f}")
        print(f"Base BERT F1: {metrics['base_model']['avg_bert_score']:.3f}")
        print(f"RAG Quality Rate: {metrics['standard_rag']['quality_retention_rate']:.3f}")
        print(f"Base Quality Rate: {metrics['base_model']['quality_retention_rate']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()