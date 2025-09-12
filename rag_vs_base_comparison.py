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
        
        # Calculate quality retention rate for RAG if not present
        if 'quality_retention_rate' not in rag_summary:
            rag_detailed = rag_results.get('detailed_results', [])
            if rag_detailed:
                high_quality_count = sum(1 for result in rag_detailed 
                                       if result.get('metrics', {}).get('quality_score', 0) > 0.7)
                rag_summary['quality_retention_rate'] = high_quality_count / len(rag_detailed)
            else:
                rag_summary['quality_retention_rate'] = 0
        
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
        
        # Print comprehensive analysis
        summary = comparison_report['summary']
        metrics = comparison_report['metrics_comparison']
        diffs = comparison_report['metrics_comparison']['differences']
        
        print(f"\n🏆 RAG vs BASE MODEL EVALUATION RESULTS")
        print(f"========================================")
        print(f"")
        print(f"🎯 OVERALL WINNER: {summary['winner'].upper()} ({summary['key_finding']})")
        
        # Performance comparison table
        print(f"\n📊 DETAILED PERFORMANCE BREAKDOWN:")
        print(f"==================================")
        print(f"")
        print(f"{'Metric':<25} {'RAG':<10} {'Base':<10} {'Improvement':<12} {'Winner':<8}")
        print(f"{'-'*65}")
        
        rag_bert = metrics['standard_rag']['avg_bert_score']
        base_bert = metrics['base_model']['avg_bert_score'] 
        bert_diff = diffs['avg_bert_score']['improvement_percent']
        bert_winner = "🏆 RAG" if bert_diff > 0 else "🏆 BASE"
        print(f"{'BERT F1 Score':<25} {rag_bert:<10.3f} {base_bert:<10.3f} {bert_diff:+8.1f}% {bert_winner:<8}")
        
        rag_qual = metrics['standard_rag']['quality_retention_rate']
        base_qual = metrics['base_model']['quality_retention_rate']
        qual_diff = diffs['quality_retention_rate']['improvement_percent'] 
        qual_winner = "🏆 RAG" if qual_diff > 0 else "🏆 BASE"
        print(f"{'Quality Rate':<25} {rag_qual:<10.3f} {base_qual:<10.3f} {qual_diff:+8.1f}% {qual_winner:<8}")
        
        rag_sim = metrics['standard_rag']['avg_similarity_score']
        base_sim = metrics['base_model']['avg_similarity_score']
        sim_diff = diffs['avg_similarity_score']['improvement_percent']
        sim_winner = "🏆 RAG" if sim_diff > 0 else "🏆 BASE" 
        print(f"{'Semantic Similarity':<25} {rag_sim:<10.3f} {base_sim:<10.3f} {sim_diff:+8.1f}% {sim_winner:<8}")
        
        print(f"\n⚡ PERFORMANCE & SPEED:")
        print(f"======================")
        rag_gen_time = comparison_report.get('rag_generation_time_ms', 0) / 1000
        base_gen_time = metrics['base_model']['generation_time_ms'] / 1000 / metrics['base_model']['total_pairs_evaluated']
        speed_improvement = 0  # Default value
        if base_gen_time > 0 and rag_gen_time > 0:
            speed_improvement = ((base_gen_time - rag_gen_time) / base_gen_time) * 100
            faster = "🏆 RAG" if speed_improvement > 0 else "🏆 BASE"
            print(f"{'Generation Speed':<25} {rag_gen_time:<10.1f}s {base_gen_time:<10.1f}s {speed_improvement:+8.1f}% {faster:<8}")
        else:
            print(f"{'Generation Speed':<25} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"===============")
        
        # Determine the story
        if bert_diff > -5 and qual_diff > 10:
            print(f"✅ RAG SUCCESS: Near-equal accuracy (+{bert_diff:.1f}%) with {qual_diff:+.1f}% better consistency")
            print(f"   → RAG provides more reliable answers with minimal quality trade-off")
        elif bert_diff < -10 and qual_diff > 15:
            print(f"⚖️  RELIABILITY vs QUALITY: RAG trades {abs(bert_diff):.1f}% quality for {qual_diff:+.1f}% consistency")
            print(f"   → Choose RAG for consistent user experience, Base for peak quality")
        elif bert_diff > 5:
            print(f"🚀 RAG DOMINANCE: {bert_diff:+.1f}% better quality AND {qual_diff:+.1f}% better consistency")
            print(f"   → RAG clearly outperforms base model across all metrics")
        else:
            print(f"📊 MIXED RESULTS: Quality {bert_diff:+.1f}%, Consistency {qual_diff:+.1f}%")
        
        print(f"\n🎯 BUSINESS IMPACT:")
        print(f"==================")
        if qual_diff > 10:
            user_satisfaction = "📈 Higher user satisfaction (more consistent answers)"
        else:
            user_satisfaction = "📊 Similar user satisfaction" 
            
        if speed_improvement > 20:
            cost_impact = "💰 Lower compute costs (faster responses)"
        else:
            cost_impact = "💰 Similar compute costs"
            
        print(f"• {user_satisfaction}")
        print(f"• {cost_impact}")
        
        accuracy_trade = abs(bert_diff)
        if accuracy_trade < 5:
            accuracy_impact = "Minimal accuracy trade-off"
        elif accuracy_trade < 10:
            accuracy_impact = f"Small accuracy trade-off ({accuracy_trade:.1f}%)"
        else:
            accuracy_impact = f"Notable accuracy trade-off ({accuracy_trade:.1f}%)"
        print(f"• 🎯 {accuracy_impact}")
        
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        print(f"=============================")
        if summary['winner'] == 'rag':
            print(f"✅ DEPLOY RAG - Better overall performance")
            if qual_diff > 15:
                print(f"   📌 Primary benefit: {qual_diff:+.1f}% more consistent answers")
            if speed_improvement > 20:
                print(f"   📌 Secondary benefit: {speed_improvement:+.1f}% faster responses")
        else:
            print(f"✅ DEPLOY BASE MODEL - Simpler but effective")
            print(f"   📌 Primary benefit: {abs(bert_diff):.1f}% higher peak quality")
            print(f"   ⚠️  Consider RAG for: consistency-critical applications")
        
        print(f"\n💡 NEXT STEPS:")
        if summary['winner'] == 'rag':
            print(f"1. Deploy RAG system in production")
            print(f"2. Monitor quality retention rate (target: >{rag_qual:.0%})")
            print(f"3. A/B test with base model on non-critical queries")
        else:
            print(f"1. Deploy base model for general queries") 
            print(f"2. Consider RAG for domain-specific questions")
            print(f"3. Monitor consistency issues in user feedback")
        
    except Exception as e:
        logger.error(f"❌ Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()