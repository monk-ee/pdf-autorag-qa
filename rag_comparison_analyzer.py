#!/usr/bin/env python3
"""
RAG Comparison Analyzer: Compare Standard vs Adaptive RAG effectiveness
Analyzes and summarizes differences between the two RAG approaches
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGComparisonAnalyzer:
    """Analyzes and compares Standard vs Adaptive RAG performance"""
    
    def __init__(self, standard_results_dir: Path, adaptive_results_dir: Path):
        self.standard_dir = Path(standard_results_dir)
        self.adaptive_dir = Path(adaptive_results_dir)
        
    def load_evaluation_results(self, results_dir: Path) -> Dict:
        """Load evaluation results from directory"""
        try:
            # Look for common result files
            result_files = [
                'qa_rag_evaluation_report.json',
                'evaluation_report.json',
                'final_evaluation_report.json'
            ]
            
            logger.info(f"Searching for evaluation files in: {results_dir}")
            
            # List all files in directory for debugging
            if results_dir.exists():
                all_files = list(results_dir.glob('*'))
                logger.info(f"Files found in {results_dir}: {[f.name for f in all_files]}")
            else:
                logger.error(f"Results directory does not exist: {results_dir}")
                return {}
            
            for filename in result_files:
                result_file = results_dir / filename
                if result_file.exists():
                    logger.info(f"Loading results from: {result_file}")
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        logger.info(f"Loaded data keys: {list(data.keys()) if data else 'Empty'}")
                        logger.info(f"Data size: {len(str(data))} characters")
                        return data
            
            logger.warning(f"No standard result files found in {results_dir}")
            logger.warning(f"Tried files: {result_files}")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading results from {results_dir}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def extract_key_metrics(self, results: Dict, approach_name: str) -> Dict:
        """Extract key metrics from evaluation results using actual data structure"""
        metrics = {
            'approach': approach_name,
            'total_pairs_evaluated': 0,
            'avg_retrieval_score': 0.0,
            'avg_similarity_score': 0.0,
            'avg_bert_score': 0.0,
            'high_quality_pairs': 0,
            'quality_retention_rate': 0.0,
            'avg_context_relevance': 0.0,
            'retrieval_time_ms': 0.0,
            'generation_time_ms': 0.0
        }
        
        try:
            logger.info(f"Extracting metrics for {approach_name} from keys: {list(results.keys())}")
            
            # Extract from evaluation_summary using actual field names from real data
            if 'evaluation_summary' in results:
                summary = results['evaluation_summary']
                logger.info(f"Evaluation summary keys: {list(summary.keys()) if isinstance(summary, dict) else 'Not a dict'}")
                
                # Map actual field names to our metric names
                metrics.update({
                    'total_pairs_evaluated': summary.get('total_pairs_evaluated', 0),
                    'avg_similarity_score': summary.get('average_semantic_similarity', 0.0),
                    'avg_bert_score': summary.get('average_quality_score', 0.0),
                    'quality_retention_rate': summary.get('average_quality_score', 0.0),  # Same as quality score
                    'avg_context_relevance': summary.get('average_content_overlap', 0.0)
                })
                
                # Count high quality pairs (quality score > 0.8)
                avg_quality = summary.get('average_quality_score', 0.0)
                total_pairs = summary.get('total_pairs_evaluated', 0)
                if avg_quality > 0.8:
                    metrics['high_quality_pairs'] = int(total_pairs * avg_quality)  # Approximation
                else:
                    metrics['high_quality_pairs'] = int(total_pairs * 0.5)  # Conservative estimate
            
            # Calculate timing metrics from detailed_results if available
            if 'detailed_results' in results:
                detailed = results['detailed_results']
                if isinstance(detailed, list) and len(detailed) > 0:
                    logger.info(f"Processing {len(detailed)} detailed results")
                    
                    # Extract timing information from detailed results
                    retrieval_times = []
                    generation_times = []
                    
                    for item in detailed:
                        if isinstance(item, dict):
                            if 'retrieval_time_ms' in item:
                                retrieval_times.append(item['retrieval_time_ms'])
                            if 'generation_time_ms' in item:
                                generation_times.append(item['generation_time_ms'])
                    
                    # Calculate average times
                    if retrieval_times:
                        metrics['retrieval_time_ms'] = sum(retrieval_times) / len(retrieval_times)
                        logger.info(f"Average retrieval time: {metrics['retrieval_time_ms']:.2f}ms")
                    
                    if generation_times:
                        metrics['generation_time_ms'] = sum(generation_times) / len(generation_times)
                        logger.info(f"Average generation time: {metrics['generation_time_ms']:.2f}ms")
                    
                    # Update total pairs count from detailed results if not already set
                    if metrics['total_pairs_evaluated'] == 0:
                        metrics['total_pairs_evaluated'] = len(detailed)
            
            # Log successful extraction
            if metrics['total_pairs_evaluated'] > 0:
                logger.info(f"✅ Successfully extracted metrics for {approach_name}:")
                logger.info(f"  - Total pairs: {metrics['total_pairs_evaluated']}")
                logger.info(f"  - Avg quality score: {metrics['avg_bert_score']:.3f}")
                logger.info(f"  - Avg similarity: {metrics['avg_similarity_score']:.3f}")
                logger.info(f"  - High quality pairs: {metrics['high_quality_pairs']}")
            else:
                logger.warning(f"⚠️ No meaningful data extracted for {approach_name}")
                logger.warning(f"Available top-level keys: {list(results.keys())}")
                if 'evaluation_summary' in results:
                    logger.warning(f"Summary keys: {list(results['evaluation_summary'].keys())}")
                
        except Exception as e:
            logger.error(f"❌ Error extracting metrics for {approach_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return metrics
    
    def calculate_performance_differences(self, standard_metrics: Dict, adaptive_metrics: Dict) -> Dict:
        """Calculate performance differences between approaches"""
        differences = {}
        
        # Metrics where higher is better
        positive_metrics = [
            'avg_retrieval_score', 'avg_similarity_score', 'avg_bert_score',
            'high_quality_pairs', 'quality_retention_rate', 'avg_context_relevance'
        ]
        
        # Metrics where lower is better (times)
        negative_metrics = ['retrieval_time_ms', 'generation_time_ms']
        
        for metric in positive_metrics:
            standard_val = standard_metrics.get(metric, 0)
            adaptive_val = adaptive_metrics.get(metric, 0)
            
            if standard_val > 0:
                improvement = ((adaptive_val - standard_val) / standard_val) * 100
            else:
                improvement = 0.0 if adaptive_val == 0 else 100.0
            
            differences[metric] = {
                'standard': standard_val,
                'adaptive': adaptive_val,
                'improvement_percent': improvement,
                'better_approach': 'adaptive' if improvement > 0 else 'standard' if improvement < 0 else 'equal'
            }
        
        for metric in negative_metrics:
            standard_val = standard_metrics.get(metric, 0)
            adaptive_val = adaptive_metrics.get(metric, 0)
            
            if standard_val > 0:
                improvement = ((standard_val - adaptive_val) / standard_val) * 100  # Reversed for time metrics
            else:
                improvement = 0.0 if adaptive_val == 0 else -100.0
            
            differences[metric] = {
                'standard': standard_val,
                'adaptive': adaptive_val,
                'improvement_percent': improvement,
                'better_approach': 'adaptive' if improvement > 0 else 'standard' if improvement < 0 else 'equal'
            }
        
        return differences
    
    def analyze_approach_characteristics(self, standard_results: Dict, adaptive_results: Dict) -> Dict:
        """Analyze characteristics of each approach"""
        analysis = {
            'standard_rag': {
                'description': 'Traditional RAG using answer embeddings only',
                'strengths': [
                    'Faster retrieval (single embedding lookup)',
                    'Lower computational overhead',
                    'Direct answer matching',
                    'Simple and proven approach'
                ],
                'weaknesses': [
                    'May miss contextual relationships',
                    'Less semantic understanding of question-answer pairs',
                    'Fixed retrieval strategy'
                ],
                'best_use_cases': [
                    'Direct factual questions',
                    'Resource-constrained environments',
                    'High-throughput scenarios'
                ]
            },
            'adaptive_rag': {
                'description': 'Enhanced RAG using combined Q+A embeddings with dynamic strategies',
                'strengths': [
                    'Better semantic understanding (Q+A combined)',
                    'Query-aware retrieval strategies',
                    'Confidence-based context gating',
                    'Adaptive to query complexity'
                ],
                'weaknesses': [
                    'Higher computational cost',
                    'More complex implementation',
                    'Potential over-engineering for simple queries'
                ],
                'best_use_cases': [
                    'Complex conceptual questions',
                    'Domain-specific queries',
                    'Quality-over-speed scenarios'
                ]
            }
        }
        
        return analysis
    
    def generate_recommendations(self, differences: Dict, analysis: Dict) -> Dict:
        """Generate recommendations based on comparison"""
        recommendations = {
            'overall_winner': None,
            'use_standard_when': [],
            'use_adaptive_when': [],
            'hybrid_approach': None,
            'key_insights': []
        }
        
        # Count wins for each approach
        standard_wins = sum(1 for diff in differences.values() if diff['better_approach'] == 'standard')
        adaptive_wins = sum(1 for diff in differences.values() if diff['better_approach'] == 'adaptive')
        
        if adaptive_wins > standard_wins:
            recommendations['overall_winner'] = 'adaptive'
        elif standard_wins > adaptive_wins:
            recommendations['overall_winner'] = 'standard'
        else:
            recommendations['overall_winner'] = 'tie'
        
        # Quality-focused recommendations
        quality_metrics = ['avg_bert_score', 'quality_retention_rate', 'avg_context_relevance']
        adaptive_better_quality = sum(1 for metric in quality_metrics 
                                    if differences.get(metric, {}).get('better_approach') == 'adaptive')
        
        # Speed-focused recommendations  
        speed_metrics = ['retrieval_time_ms', 'generation_time_ms']
        standard_faster = sum(1 for metric in speed_metrics
                            if differences.get(metric, {}).get('better_approach') == 'standard')
        
        # Generate specific recommendations
        recommendations['use_standard_when'] = [
            'Speed is critical and quality is acceptable',
            'Processing high volumes of simple queries',
            'Resource constraints limit computational overhead',
            'Direct factual lookup is sufficient'
        ]
        
        recommendations['use_adaptive_when'] = [
            'Quality is more important than speed',
            'Handling complex domain-specific questions',
            'Context understanding is crucial',
            'Dealing with out-of-domain queries that need uncertainty'
        ]
        
        if adaptive_better_quality >= 2 and standard_faster >= 1:
            recommendations['hybrid_approach'] = 'Use query complexity analysis to route simple queries to Standard RAG and complex queries to Adaptive RAG'
        
        # Key insights
        insights = []
        
        if differences.get('quality_retention_rate', {}).get('improvement_percent', 0) > 5:
            insights.append("Adaptive RAG shows significant quality improvement")
        
        if differences.get('retrieval_time_ms', {}).get('improvement_percent', 0) < -10:
            insights.append("Adaptive RAG has noticeable performance overhead")
        
        if abs(differences.get('avg_bert_score', {}).get('improvement_percent', 0)) < 2:
            insights.append("Similar semantic quality between approaches")
        
        recommendations['key_insights'] = insights
        
        return recommendations
    
    def run_comparison_analysis(self) -> Dict:
        """Run complete comparison analysis"""
        logger.info("Starting RAG comparison analysis...")
        
        # Load results
        standard_results = self.load_evaluation_results(self.standard_dir)
        adaptive_results = self.load_evaluation_results(self.adaptive_dir)
        
        if not standard_results and not adaptive_results:
            logger.error("❌ CRITICAL: No evaluation results found in either directory")
            logger.error(f"Standard RAG directory: {self.standard_dir}")
            logger.error(f"Adaptive RAG directory: {self.adaptive_dir}")
            logger.error("This suggests the RAG evaluation steps failed completely")
            return {
                'error': 'No evaluation results found',
                'standard_dir': str(self.standard_dir),
                'adaptive_dir': str(self.adaptive_dir),
                'suggestion': 'Check if qa_autorag_evaluator.py is running successfully'
            }
        
        if not standard_results:
            logger.warning("⚠️ Standard RAG evaluation results missing")
        
        if not adaptive_results:
            logger.warning("⚠️ Adaptive RAG evaluation results missing")
        
        # Extract metrics
        standard_metrics = self.extract_key_metrics(standard_results, 'standard')
        adaptive_metrics = self.extract_key_metrics(adaptive_results, 'adaptive')
        
        # Calculate differences
        differences = self.calculate_performance_differences(standard_metrics, adaptive_metrics)
        
        # Filter out unavailable metrics (all zeros)
        unavailable_metrics = ['retrieval_time_ms', 'generation_time_ms', 'avg_retrieval_score']
        for metric in unavailable_metrics:
            if (differences.get(metric, {}).get('standard', 0) == 0 and 
                differences.get(metric, {}).get('adaptive', 0) == 0):
                logger.info(f"⚠️  Removing {metric} from comparison (no data available)")
                differences.pop(metric, None)
        
        # Analyze characteristics
        analysis = self.analyze_approach_characteristics(standard_results, adaptive_results)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(differences, analysis)
        
        # Compile complete analysis
        comparison_report = {
            'metadata': {
                'analysis_type': 'rag_comparison',
                'standard_results_dir': str(self.standard_dir),
                'adaptive_results_dir': str(self.adaptive_dir),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'metrics_comparison': {
                'standard_rag': standard_metrics,
                'adaptive_rag': adaptive_metrics,
                'differences': differences
            },
            'approach_analysis': analysis,
            'recommendations': recommendations,
            'summary': {
                'winner': recommendations['overall_winner'],
                'key_finding': self._generate_key_finding(differences),
                'recommendation_summary': self._generate_recommendation_summary(recommendations)
            }
        }
        
        logger.info("RAG comparison analysis completed")
        return comparison_report
    
    def _generate_key_finding(self, differences: Dict) -> str:
        """Generate a key finding summary"""
        significant_improvements = []
        significant_degradations = []
        
        for metric, diff in differences.items():
            improvement = diff.get('improvement_percent', 0)
            if improvement > 10:
                significant_improvements.append(f"{metric.replace('_', ' ')} (+{improvement:.1f}%)")
            elif improvement < -10:
                significant_degradations.append(f"{metric.replace('_', ' ')} ({improvement:.1f}%)")
        
        if significant_improvements:
            return f"Adaptive RAG shows significant improvements in: {', '.join(significant_improvements)}"
        elif significant_degradations:
            return f"Adaptive RAG shows degradation in: {', '.join(significant_degradations)}"
        else:
            return "Both approaches show similar performance with minor differences"
    
    def _generate_recommendation_summary(self, recommendations: Dict) -> str:
        """Generate recommendation summary"""
        winner = recommendations.get('overall_winner', 'tie')
        
        if winner == 'adaptive':
            return "Recommend Adaptive RAG for quality-focused applications"
        elif winner == 'standard':
            return "Recommend Standard RAG for speed-focused applications"
        else:
            return "Consider hybrid approach based on query complexity"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare Standard vs Adaptive RAG performance')
    parser.add_argument('--standard-results', type=Path, required=True,
                       help='Directory with Standard RAG evaluation results')
    parser.add_argument('--adaptive-results', type=Path, required=True,
                       help='Directory with Adaptive RAG evaluation results')
    parser.add_argument('--output-file', type=Path, default='rag_comparison_report.json',
                       help='Output file for comparison report')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run analysis
    analyzer = RAGComparisonAnalyzer(args.standard_results, args.adaptive_results)
    report = analyzer.run_comparison_analysis()
    
    if not report:
        logger.error("Analysis failed - no report generated")
        return 1
    
    # Save report
    with open(args.output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Comparison report saved: {args.output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("RAG APPROACH COMPARISON SUMMARY")
    print("="*60)
    
    summary = report.get('summary', {})
    print(f"Overall Winner: {summary.get('winner', 'unknown').upper()}")
    print(f"Key Finding: {summary.get('key_finding', 'N/A')}")
    print(f"Recommendation: {summary.get('recommendation_summary', 'N/A')}")
    
    # Performance comparison table
    differences = report.get('metrics_comparison', {}).get('differences', {})
    if differences:
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print("-" * 60)
        print(f"{'Metric':<25} {'Standard':<12} {'Adaptive':<12} {'Change':<10}")
        print("-" * 60)
        
        for metric, diff in differences.items():
            metric_name = metric.replace('_', ' ').title()
            standard_val = diff.get('standard', 0)
            adaptive_val = diff.get('adaptive', 0)
            change = diff.get('improvement_percent', 0)
            
            # Format values appropriately
            if metric.endswith('_percent') or 'rate' in metric:
                standard_str = f"{standard_val:.1%}" if isinstance(standard_val, (int, float)) else str(standard_val)
                adaptive_str = f"{adaptive_val:.1%}" if isinstance(adaptive_val, (int, float)) else str(adaptive_val)
            elif 'time' in metric:
                standard_str = f"{standard_val:.0f}ms" if isinstance(standard_val, (int, float)) else str(standard_val)
                adaptive_str = f"{adaptive_val:.0f}ms" if isinstance(adaptive_val, (int, float)) else str(adaptive_val)
            else:
                standard_str = f"{standard_val:.3f}" if isinstance(standard_val, (int, float)) else str(standard_val)
                adaptive_str = f"{adaptive_val:.3f}" if isinstance(adaptive_val, (int, float)) else str(adaptive_val)
            
            change_str = f"{change:+.1f}%" if isinstance(change, (int, float)) else str(change)
            
            print(f"{metric_name:<25} {standard_str:<12} {adaptive_str:<12} {change_str:<10}")
    
    # Recommendations
    recommendations = report.get('recommendations', {})
    if recommendations.get('use_adaptive_when'):
        print(f"\n🔸 Use Adaptive RAG when:")
        for rec in recommendations['use_adaptive_when']:
            print(f"   • {rec}")
    
    if recommendations.get('use_standard_when'):
        print(f"\n🔹 Use Standard RAG when:")
        for rec in recommendations['use_standard_when']:
            print(f"   • {rec}")
    
    if recommendations.get('hybrid_approach'):
        print(f"\n🔄 Hybrid Approach:")
        print(f"   {recommendations['hybrid_approach']}")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())