"""
Performance Analytics

Advanced performance analysis and metrics calculation for Q&A extraction/generation.
"""
import statistics
from collections import Counter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Advanced performance analysis for Q&A extraction and generation."""
    
    @staticmethod
    def analyze_content_quality(all_pairs: List[Dict[str, str]], 
                              parsing_stats_list: List[Dict]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of extracted Q&A pairs."""
        if not all_pairs:
            return {"error": "No Q&A pairs to analyze"}
        
        # Extract lengths and word counts
        question_lengths = [pair['metadata']['question_length'] for pair in all_pairs]
        answer_lengths = [pair['metadata']['answer_length'] for pair in all_pairs]
        question_words = [pair['metadata']['question_words'] for pair in all_pairs]
        answer_words = [pair['metadata']['answer_words'] for pair in all_pairs]
        
        # Analyze question patterns
        questions = [pair['instruction'] for pair in all_pairs]
        question_starters = [q.split()[0].lower() if q.split() else "" for q in questions]
        question_starter_counts = Counter(question_starters)
        
        # Calculate parsing efficiency
        total_parsing_stats = {
            'total_raw_responses': len(parsing_stats_list),
            'total_q_lines': sum(s['q_lines'] for s in parsing_stats_list),
            'total_a_lines': sum(s['a_lines'] for s in parsing_stats_list),
            'total_unparseable_lines': sum(s['unparseable_lines'] for s in parsing_stats_list),
            'total_incomplete_pairs': sum(s['incomplete_pairs'] for s in parsing_stats_list),
            'parsing_success_rate': len(all_pairs) / max(1, sum(s['q_lines'] for s in parsing_stats_list))
        }
        
        return {
            'content_statistics': {
                'total_pairs': len(all_pairs),
                'question_length_stats': {
                    'mean': statistics.mean(question_lengths),
                    'median': statistics.median(question_lengths),
                    'min': min(question_lengths),
                    'max': max(question_lengths),
                    'std_dev': statistics.stdev(question_lengths) if len(question_lengths) > 1 else 0
                },
                'answer_length_stats': {
                    'mean': statistics.mean(answer_lengths),
                    'median': statistics.median(answer_lengths),
                    'min': min(answer_lengths),
                    'max': max(answer_lengths),
                    'std_dev': statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0
                },
                'word_count_stats': {
                    'avg_question_words': statistics.mean(question_words),
                    'avg_answer_words': statistics.mean(answer_words),
                    'question_word_range': (min(question_words), max(question_words)),
                    'answer_word_range': (min(answer_words), max(answer_words))
                }
            },
            'quality_metrics': {
                'avg_answer_to_question_ratio': statistics.mean(answer_lengths) / statistics.mean(question_lengths),
                'question_diversity_score': len(set(questions)) / len(questions),
                'common_question_starters': dict(question_starter_counts.most_common(10)),
                'empty_answers': sum(1 for pair in all_pairs if not pair['output'].strip()),
                'very_short_answers': sum(1 for pair in all_pairs if len(pair['output'].split()) < 3),
                'very_long_answers': sum(1 for pair in all_pairs if len(pair['output'].split()) > 50)
            },
            'parsing_analysis': total_parsing_stats
        }
    
    @staticmethod
    def estimate_economic_metrics(total_time: float, num_pairs: int, gpu_used: bool) -> Dict[str, Any]:
        """Estimate costs and environmental impact."""
        # L40S pricing estimates (adjust based on actual provider pricing)
        l40s_cost_per_hour = 2.50  # Approximate spot pricing
        a10g_cost_per_hour = 1.20  # Alternative GPU option
        cpu_cost_per_hour = 0.30   # CPU-only baseline
        
        # Power consumption estimates (watts)
        l40s_power_watts = 300
        a10g_power_watts = 200
        cpu_power_watts = 100
        
        # Carbon intensity (kg CO2 per kWh) - global average
        carbon_intensity = 0.5
        
        runtime_hours = total_time / 3600
        
        if gpu_used:
            estimated_cost = runtime_hours * l40s_cost_per_hour
            power_usage_kwh = (l40s_power_watts * runtime_hours) / 1000
            carbon_footprint_kg = power_usage_kwh * carbon_intensity
        else:
            estimated_cost = runtime_hours * cpu_cost_per_hour
            power_usage_kwh = (cpu_power_watts * runtime_hours) / 1000
            carbon_footprint_kg = power_usage_kwh * carbon_intensity
        
        return {
            'cost_estimates': {
                'runtime_hours': runtime_hours,
                'estimated_cost_usd': estimated_cost,
                'cost_per_qa_pair': estimated_cost / num_pairs if num_pairs > 0 else 0,
                'cost_per_1000_pairs': (estimated_cost / num_pairs) * 1000 if num_pairs > 0 else 0,
                'projected_cost_1m_pairs': (estimated_cost / num_pairs) * 1000000 if num_pairs > 0 else 0
            },
            'environmental_impact': {
                'power_consumption_kwh': power_usage_kwh,
                'carbon_footprint_kg_co2': carbon_footprint_kg,
                'carbon_per_1000_pairs_g': (carbon_footprint_kg * 1000 / num_pairs) if num_pairs > 0 else 0
            },
            'scaling_projections': {
                'pairs_per_dollar': num_pairs / estimated_cost if estimated_cost > 0 else 0,
                'pairs_per_kwh': num_pairs / power_usage_kwh if power_usage_kwh > 0 else 0,
                'breakeven_vs_cpu_pairs': None  # Would need CPU benchmark
            }
        }
    
    @staticmethod
    def calculate_batch_consistency(batch_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate batch processing consistency metrics."""
        if not batch_stats:
            return {}
        
        batch_times = [b.get('total_batch_time', 0) for b in batch_stats]
        pairs_per_batch = [b.get('qa_pairs_extracted', b.get('qa_pairs_generated', 0)) for b in batch_stats]
        inference_times = [b.get('inference_time', 0) for b in batch_stats]
        
        return {
            'timing_variance': statistics.stdev(batch_times) if len(batch_times) > 1 else 0,
            'pairs_per_batch_variance': statistics.stdev(pairs_per_batch) if len(pairs_per_batch) > 1 else 0,
            'inference_time_variance': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'consistency_score': 1 - (statistics.stdev(batch_times) / statistics.mean(batch_times)) if batch_times and statistics.mean(batch_times) > 0 else 0,
            'avg_batch_time': statistics.mean(batch_times) if batch_times else 0,
            'avg_pairs_per_batch': statistics.mean(pairs_per_batch) if pairs_per_batch else 0
        }
    
    @staticmethod
    def analyze_pipeline_stages(total_time: float, total_inference_time: float, 
                              batch_stats: List[Dict]) -> Dict[str, Any]:
        """Analyze performance across different pipeline stages."""
        total_parse_time = sum(b.get('parse_time', 0) for b in batch_stats)
        
        pipeline_analysis = {
            'stage_breakdown': {
                'model_inference_percent': (total_inference_time / total_time) * 100 if total_time > 0 else 0,
                'parsing_percent': (total_parse_time / total_time) * 100 if total_time > 0 else 0,
                'io_overhead_percent': ((total_time - total_inference_time - total_parse_time) / total_time) * 100 if total_time > 0 else 0
            },
            'bottleneck_analysis': {
                'primary_bottleneck': 'inference' if total_inference_time > total_parse_time else 'parsing',
                'parallelization_effectiveness': None  # Would need CPU count
            }
        }
        
        return pipeline_analysis
    
    @staticmethod
    def generate_comprehensive_report(all_pairs: List[Dict], parsing_stats_list: List[Dict],
                                    performance_metrics: Dict, batch_stats: List[Dict]) -> Dict[str, Any]:
        """Generate a comprehensive performance analysis report."""
        content_analysis = PerformanceAnalyzer.analyze_content_quality(all_pairs, parsing_stats_list)
        
        total_time = performance_metrics.get('processing_stats', {}).get('total_runtime', 0)
        total_inference_time = performance_metrics.get('processing_stats', {}).get('model_inference_time', 0)
        gpu_used = performance_metrics.get('model_name', '').lower() in ['cuda', 'gpu'] or 'gpu' in str(performance_metrics.get('resource_utilization', {}))
        
        economic_analysis = PerformanceAnalyzer.estimate_economic_metrics(
            total_time, len(all_pairs), gpu_used
        )
        
        batch_consistency = PerformanceAnalyzer.calculate_batch_consistency(batch_stats)
        pipeline_analysis = PerformanceAnalyzer.analyze_pipeline_stages(
            total_time, total_inference_time, batch_stats
        )
        
        return {
            'content_quality_analysis': content_analysis,
            'economic_analysis': economic_analysis,
            'batch_consistency': batch_consistency,
            'pipeline_efficiency': pipeline_analysis,
            'recommendations': PerformanceAnalyzer._generate_recommendations(
                content_analysis, economic_analysis, batch_consistency, pipeline_analysis
            )
        }
    
    @staticmethod
    def _generate_recommendations(content_analysis: Dict, economic_analysis: Dict,
                                batch_consistency: Dict, pipeline_analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Content quality recommendations
        if content_analysis.get('quality_metrics', {}).get('parsing_success_rate', 0) < 0.8:
            recommendations.append("Consider improving prompt engineering - parsing success rate is below 80%")
        
        # Economic recommendations
        cost_per_pair = economic_analysis.get('cost_estimates', {}).get('cost_per_qa_pair', 0)
        if cost_per_pair > 0.01:  # More than 1 cent per pair
            recommendations.append("Consider optimizing batch size or using smaller model - cost per pair is high")
        
        # Batch consistency recommendations
        consistency_score = batch_consistency.get('consistency_score', 0)
        if consistency_score < 0.8:
            recommendations.append("Batch processing is inconsistent - consider adjusting batch size or system resources")
        
        # Pipeline efficiency recommendations
        inference_percent = pipeline_analysis.get('stage_breakdown', {}).get('model_inference_percent', 0)
        if inference_percent < 60:
            recommendations.append("Model inference time is low - consider increasing batch size for better GPU utilization")
        
        return recommendations