#!/usr/bin/env python3
"""
Q&A Pair Selection Script for AutoRAG Pipeline

Selects top K highest-quality Q&A pairs from matrix generation results.
Replaces inline Python in GitHub Actions workflows for maintainability.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from contextlib import nullcontext

# Import pipeline debugging utilities
try:
    from pipeline_debug import create_pipeline_debugger
    DEBUGGING_AVAILABLE = True
except ImportError:
    DEBUGGING_AVAILABLE = False


def quality_score(pair: Dict[str, Any], debugger=None) -> float:
    """
    Calculate quality score for a Q&A pair using heuristics.
    FIXED: Now handles both HF format (instruction/output) and QA format (question/answer)
    
    Args:
        pair: Q&A pair dictionary with 'instruction'/'output' or 'question'/'answer' keys
        debugger: Optional debugger for logging
        
    Returns:
        Quality score (0.0-1.0, higher is better)
    """
    # Handle both HF format (instruction/output) and QA format (question/answer)
    question = pair.get('instruction', pair.get('question', ''))
    answer = pair.get('output', pair.get('answer', ''))
    
    if debugger:
        debugger.logger.debug(f"Scoring pair: Q='{question[:50]}...', A='{answer[:50]}...'")
    
    if not question.strip() or not answer.strip():
        if debugger:
            debugger.logger.warning(f"Empty question or answer found: Q='{question}', A='{answer}'")
        return 0.0
    
    # Basic quality heuristics
    q_words = len(question.split())
    a_words = len(answer.split())
    
    # Base score from length (normalized)
    base_score = 0.3
    
    # Question length bonus (sweet spot: 10-30 words)
    if q_words >= 8:
        base_score += 0.2
    if q_words >= 15:
        base_score += 0.1
    if q_words > 50:  # Penalty for too long
        base_score -= 0.1
    
    # Answer length bonus (sweet spot: 20-100 words)
    if a_words >= 15:
        base_score += 0.2
    if a_words >= 30:
        base_score += 0.1
    if a_words >= 50:
        base_score += 0.1
    if a_words > 200:  # Penalty for too long
        base_score -= 0.1
    
    # Question quality indicators
    question_lower = question.lower()
    has_good_keywords = any(word in question_lower 
                           for word in ['how', 'why', 'what', 'explain', 'describe', 
                                       'compare', 'analyze', 'evaluate', 'when', 'where'])
    if has_good_keywords:
        base_score += 0.1
    
    # Answer quality indicators
    answer_lower = answer.lower()
    has_explanation = any(phrase in answer_lower 
                         for phrase in ['because', 'therefore', 'this means', 'as a result',
                                       'for example', 'such as', 'in order to'])
    if has_explanation:
        base_score += 0.1
    
    # Penalties for poor quality
    if a_words < 10:  # Very short answers
        base_score *= 0.5
    
    if question.count('?') == 0 and any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where']):
        base_score *= 0.9  # Questions should usually have question marks
    
    # Check for metadata quality score if available
    metadata_score = pair.get('metadata', {}).get('quality_score')
    if metadata_score is not None:
        # Blend with metadata score (70% our score, 30% metadata)
        base_score = base_score * 0.7 + float(metadata_score) * 0.3
    
    final_score = max(0.0, min(1.0, base_score))  # Clamp to [0, 1]
    
    if debugger:
        debugger.logger.debug(f"Final score: {final_score:.3f} (Q:{q_words}w, A:{a_words}w)")
    
    return final_score


def collect_qa_pairs(qa_artifacts_dir: Path, debugger=None) -> List[Dict[str, Any]]:
    """
    Collect all Q&A pairs from matrix generation artifacts.
    FIXED: Now handles missing directories and various file formats
    
    Args:
        qa_artifacts_dir: Directory containing JSONL files from matrix jobs
        debugger: Optional debugger for logging
        
    Returns:
        List of Q&A pairs with metadata
    """
    all_pairs = []
    
    if not qa_artifacts_dir.exists():
        if debugger:
            debugger.logger.error(f"Artifacts directory does not exist: {qa_artifacts_dir}")
            # Try alternative locations
            alternatives = [
                Path('examples/pdf-qa-generation-results'),
                Path('examples/qa-generation-advanced-balanced'),
                Path('.')  # Current directory
            ]
            for alt_dir in alternatives:
                if alt_dir.exists():
                    jsonl_files = list(alt_dir.glob('*.jsonl'))
                    if jsonl_files:
                        debugger.logger.info(f"Found alternative directory with {len(jsonl_files)} JSONL files: {alt_dir}")
                        qa_artifacts_dir = alt_dir
                        break
        else:
            print(f'❌ Artifacts directory does not exist: {qa_artifacts_dir}')
            return all_pairs
    
    jsonl_files = list(qa_artifacts_dir.glob('*.jsonl'))
    
    if debugger:
        debugger.logger.info(f'🔍 Found {len(jsonl_files)} JSONL files in {qa_artifacts_dir}')
        debugger.log_data_sample([f.name for f in jsonl_files], "JSONL files")
    
    for jsonl_file in jsonl_files:
        if debugger:
            debugger.logger.info(f'Processing {jsonl_file.name}...')
        else:
            print(f'Processing {jsonl_file.name}...')
        
        file_pairs = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Add metadata
                    data['source_file'] = jsonl_file.name
                    data['line_number'] = line_num
                    
                    # Extract matrix combination from filename
                    # Handle formats like: pdf_qa_basic_conservative_20250809_000239.jsonl
                    filename_parts = jsonl_file.stem.split('_')
                    if len(filename_parts) >= 4:
                        difficulty = filename_parts[2]  # basic/intermediate/advanced  
                        style = '_'.join(filename_parts[3:-1])  # conservative, high_creativity, etc
                        data['matrix_combination'] = f"{difficulty}_{style}"
                    else:
                        data['matrix_combination'] = jsonl_file.stem
                    
                    file_pairs.append(data)
                    
                except json.JSONDecodeError as e:
                    if debugger:
                        debugger.logger.warning(f'Skipped line {line_num} in {jsonl_file.name}: {e}')
                    else:
                        print(f'  Warning: Skipped line {line_num} in {jsonl_file.name}: {e}')
                except Exception as e:
                    if debugger:
                        debugger.logger.warning(f'Error processing line {line_num} in {jsonl_file.name}: {e}')
                    else:
                        print(f'  Warning: Error processing line {line_num} in {jsonl_file.name}: {e}')
        
        all_pairs.extend(file_pairs)
        
        if debugger:
            debugger.logger.info(f'  ✅ Loaded {len(file_pairs)} pairs from {jsonl_file.name}')
        
        if debugger and file_pairs:
            debugger.log_data_sample(file_pairs, f"Sample from {jsonl_file.name}", max_items=2)
    
    if debugger:
        debugger.logger.info(f'📊 Total pairs collected: {len(all_pairs)}')
        if all_pairs:
            debugger.log_statistics(all_pairs, "All collected pairs")
    
    return all_pairs


def select_top_k_pairs(all_pairs: List[Dict[str, Any]], top_k: int, debugger=None) -> List[Dict[str, Any]]:
    """
    Select top K pairs based on quality scoring.
    FIXED: Proper quality scoring and debugging
    
    Args:
        all_pairs: List of all Q&A pairs
        top_k: Number of top pairs to select
        debugger: Optional debugger for logging
        
    Returns:
        Selected pairs sorted by quality (best first)
    """
    if debugger:
        debugger.logger.info(f'📊 Total collected pairs: {len(all_pairs)}')
    else:
        print(f'📊 Total collected pairs: {len(all_pairs)}')
    
    if not all_pairs:
        if debugger:
            debugger.logger.warning('No pairs to score!')
        return []
    
    # Score all pairs with debugging
    scored_pairs = 0
    zero_score_count = 0
    
    for pair in all_pairs:
        score = quality_score(pair, debugger)
        pair['quality_score'] = score
        scored_pairs += 1
        
        if score == 0.0:
            zero_score_count += 1
    
    if debugger:
        debugger.logger.info(f'🎯 Scored {scored_pairs} pairs, {zero_score_count} got zero scores')
        if zero_score_count > 0:
            debugger.logger.warning(f'⚠️ {zero_score_count}/{scored_pairs} pairs got zero scores - check data format!')
            # Sample some zero-score pairs for debugging
            zero_pairs = [p for p in all_pairs if p['quality_score'] == 0.0][:3]
            for i, pair in enumerate(zero_pairs):
                debugger.logger.debug(f"Zero-score pair {i+1}: keys={list(pair.keys())}")
    
    # Sort by quality score (descending) and select top K
    sorted_pairs = sorted(all_pairs, key=lambda x: x['quality_score'], reverse=True)
    selected_pairs = sorted_pairs[:top_k]
    
    if debugger:
        debugger.logger.info(f'✅ Selected top {len(selected_pairs)} pairs')
        if selected_pairs:
            debugger.logger.info(f'📈 Quality score range: {selected_pairs[-1]["quality_score"]:.3f} - {selected_pairs[0]["quality_score"]:.3f}')
            debugger.log_data_sample(selected_pairs, "Selected pairs sample", max_items=3)
    else:
        print(f'✅ Selected top {len(selected_pairs)} pairs')
        if selected_pairs:
            print(f'📈 Quality score range: {selected_pairs[-1]["quality_score"]:.3f} - {selected_pairs[0]["quality_score"]:.3f}')
    
    return selected_pairs


def generate_selection_summary(all_pairs: List[Dict[str, Any]], 
                             selected_pairs: List[Dict[str, Any]], 
                             top_k_requested: int) -> Dict[str, Any]:
    """
    Generate summary statistics for the selection process.
    
    Args:
        all_pairs: All available Q&A pairs
        selected_pairs: Selected top-K pairs
        top_k_requested: Number of pairs requested
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        'total_pairs': len(all_pairs),
        'selected_pairs': len(selected_pairs),
        'top_k_requested': top_k_requested,
        'quality_score_range': {
            'min': selected_pairs[-1]['quality_score'] if selected_pairs else 0,
            'max': selected_pairs[0]['quality_score'] if selected_pairs else 0,
            'avg': sum(p['quality_score'] for p in selected_pairs) / len(selected_pairs) if selected_pairs else 0
        },
        'matrix_distribution': {}
    }
    
    # Count pairs by matrix combination
    for pair in selected_pairs:
        combo = pair['matrix_combination']
        summary['matrix_distribution'][combo] = summary['matrix_distribution'].get(combo, 0) + 1
        
    return summary


def main():
    parser = argparse.ArgumentParser(description='Select top K Q&A pairs from matrix generation results')
    parser.add_argument('--qa-artifacts-dir', type=Path, default='qa_artifacts',
                       help='Directory containing JSONL files from matrix jobs')
    parser.add_argument('--output-dir', type=Path, default='rag_input',
                       help='Output directory for selected pairs')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top pairs to select')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable comprehensive debugging')
    
    args = parser.parse_args()
    
    # Initialize debugger if requested
    debugger = None
    if args.debug and DEBUGGING_AVAILABLE:
        debug_level = "DEBUG" if args.verbose else "INFO"
        debugger = create_pipeline_debugger("qa_pair_selector", debug_level)
        debugger.logger.info(f"🔧 Debug mode enabled with level: {debug_level}")
    elif args.debug and not DEBUGGING_AVAILABLE:
        print("⚠️ Debug mode requested but pipeline_debug module not available")
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    try:
        # Collect all Q&A pairs with debugging
        with (debugger.step("collect_qa_pairs", 
                           input_files=[args.qa_artifacts_dir],
                           expected_outputs=[args.output_dir / "selected_qa_pairs.json"]) 
              if debugger else nullcontext()) as step:
            
            if debugger:
                debugger.logger.info(f'🔍 Collecting Q&A pairs from {args.qa_artifacts_dir}...')
            elif args.verbose:
                print(f'🔍 Collecting Q&A pairs from {args.qa_artifacts_dir}...')
                
            all_pairs = collect_qa_pairs(args.qa_artifacts_dir, debugger)
            
            if debugger and step:
                step.metrics['pairs_collected'] = len(all_pairs)
                step.metrics['source_directory'] = str(args.qa_artifacts_dir)
        
        if not all_pairs:
            if debugger:
                debugger.logger.error('❌ No Q&A pairs found in artifacts directory')
            else:
                print('❌ No Q&A pairs found in artifacts directory')
            return 1
        
        # Select top K pairs with debugging
        with (debugger.step("select_top_k_pairs") if debugger else nullcontext()) as step:
            if debugger:
                debugger.logger.info(f'🎯 Selecting top {args.top_k} pairs...')
            elif args.verbose:
                print(f'🎯 Selecting top {args.top_k} pairs...')
                
            selected_pairs = select_top_k_pairs(all_pairs, args.top_k, debugger)
            
            if debugger and step:
                step.metrics['top_k_requested'] = args.top_k
                step.metrics['pairs_selected'] = len(selected_pairs)
                step.metrics['selection_ratio'] = len(selected_pairs) / len(all_pairs) if all_pairs else 0
        
        # Save selected pairs
        output_file = args.output_dir / 'selected_qa_pairs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_pairs, f, indent=2, ensure_ascii=False)
        
        # Generate and save summary
        summary = generate_selection_summary(all_pairs, selected_pairs, args.top_k)
        summary_file = args.output_dir / 'selection_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
        # Save with debugging validation
        with (debugger.step("save_results", 
                           expected_outputs=[output_file, summary_file]) 
              if debugger else nullcontext()) as step:
            
            if debugger and step:
                step.metrics['output_file_size'] = output_file.stat().st_size
                step.metrics['summary_file_size'] = summary_file.stat().st_size
        
        if debugger:
            debugger.logger.info(f'💾 Saved selected pairs to: {output_file}')
            debugger.logger.info(f'💾 Saved selection summary to: {summary_file}')
            debugger.logger.info('📋 Selection complete!')
            debugger.finalize()
        elif args.verbose:
            print(f'💾 Saved selected pairs to: {output_file}')
            print(f'💾 Saved selection summary to: {summary_file}')
            print('📋 Selection complete!')
        
        return 0
        
    except Exception as e:
        if debugger:
            debugger.logger.error(f'❌ Error during QA pair selection: {e}')
            if args.verbose:
                import traceback
                debugger.logger.error(traceback.format_exc())
        else:
            print(f'❌ Error during QA pair selection: {e}')
            if args.verbose:
                import traceback
                traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())