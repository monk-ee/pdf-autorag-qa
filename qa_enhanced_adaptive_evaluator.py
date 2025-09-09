#!/usr/bin/env python3
"""
Enhanced Adaptive RAG Evaluator

Uses the enhanced adaptive_rag_pipeline.py with:
- Cross-encoder re-ranking
- Hybrid dense+sparse retrieval  
- Dynamic context windows
- Enhanced query classification

This replaces qa_autorag_evaluator.py for adaptive RAG evaluation.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import warnings
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import our enhanced adaptive pipeline
from adaptive_rag_pipeline import AdaptiveRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedAdaptiveRAGEvaluator:
    """Enhanced Adaptive RAG evaluation using the 4-improvement pipeline."""
    
    def __init__(self, 
                 qa_pairs_file: Path,
                 model_name: str,
                 domain_config_file: str = "audio_equipment_domain_questions.json",
                 use_quantization: bool = False,
                 device: str = 'auto'):
        """
        Initialize Enhanced Adaptive RAG evaluator.
        
        Args:
            qa_pairs_file: Path to Q&A pairs JSON (used as knowledge base)
            model_name: Hugging Face model name for generation
            domain_config_file: Domain configuration for enhanced features
            use_quantization: Whether to use 4-bit quantization
            device: Device for model loading ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        
        # Load Q&A pairs (knowledge base)
        logger.info(f"Loading Q&A pairs from: {qa_pairs_file}")
        with open(qa_pairs_file, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        
        # Initialize enhanced adaptive pipeline  
        logger.info("🚀 Initializing Enhanced Adaptive RAG Pipeline...")
        self.adaptive_pipeline = AdaptiveRAGPipeline(self.qa_pairs, domain_config_file)
        logger.info("✅ Enhanced pipeline initialized with 4 major improvements")
        
        # Load generation model
        logger.info(f"Loading generation model: {model_name}")
        self._setup_generation_model(model_name, use_quantization)
        
        # Initialize evaluation metrics - FORCE GPU
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)  # Force embedding model to GPU
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device - FORCE GPU on dedicated GPU instance"""
        if device == 'auto':
            # FORCE CUDA on GPU instance - no CPU fallbacks!
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "Unknown GPU"
            logger.info(f"🚀 FORCING GPU: {gpu_name} (CUDA: {torch.version.cuda})")
            logger.info(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            logger.info("💪 Dedicated GPU instance - no CPU fallbacks!")
        else:
            logger.info(f"🔧 Using specified device: {device}")
            
        return torch.device(device)
    
    def _setup_generation_model(self, model_name: str, use_quantization: bool):
        """Setup the generation model with optional quantization"""
        
        # Quantization config
        quantization_config = None
        if use_quantization and self.device.type == 'cuda':
            logger.info("🔧 Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            'dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            'device_map': 'auto' if self.device.type == 'cuda' else None,
            'trust_remote_code': True,
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        if not quantization_config and self.device.type != 'cuda':
            self.model = self.model.to(self.device)
        
        logger.info(f"✅ Model loaded: {model_name} on {self.device}")
    
    def generate_answer_with_adaptive_rag(self, question: str) -> tuple[str, Dict]:
        """
        Generate answer using Enhanced Adaptive RAG pipeline.
        
        Returns:
            tuple: (generated_answer, adaptive_metadata)
        """
        # 🚀 Use Enhanced Adaptive RAG Pipeline
        adaptive_result = self.adaptive_pipeline.process_query(question)
        
        # Get formatted context from adaptive pipeline
        context = adaptive_result['formatted_context']
        
        # Create enhanced prompt with adaptive context
        prompt = f"""Based on the following context from audio equipment Q&A, answer the question accurately and concisely.

Context:
{context}

Question: {question}
Answer:"""
        
        # Generate with model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode answer
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        # Return answer + adaptive metadata
        adaptive_metadata = {
            'strategy_used': adaptive_result['strategy_used'],
            'context_quality': adaptive_result['context_quality'],
            'num_retrieved': len(adaptive_result['retrieved_items']),
            'adaptive_adjustments': adaptive_result['adaptive_adjustments'],
            'query_analysis': {
                'type': adaptive_result['analysis'].query_type,
                'domain_relevance': adaptive_result['analysis'].domain_relevance,
                'complexity': adaptive_result['analysis'].complexity,
                'uncertainty_required': adaptive_result['analysis'].uncertainty_required
            }
        }
        
        return answer, adaptive_metadata
    
    def calculate_evaluation_metrics(self, original_answer: str, rag_answer: str, 
                                   question: str = "") -> Dict[str, float]:
        """Calculate evaluation metrics between original and RAG answers."""
        
        # Length-based metrics
        orig_len = len(original_answer.split())
        rag_len = len(rag_answer.split())
        length_ratio = rag_len / orig_len if orig_len > 0 else 0
        
        # Semantic similarity using embeddings
        if original_answer.strip() and rag_answer.strip():
            orig_embedding = self.embedding_model.encode([original_answer])
            rag_embedding = self.embedding_model.encode([rag_answer])
            semantic_similarity = float(cosine_similarity(orig_embedding, rag_embedding)[0][0])
        else:
            semantic_similarity = 0.0
        
        # Content overlap (simple word-based)
        orig_words = set(original_answer.lower().split())
        rag_words = set(rag_answer.lower().split())
        
        if len(orig_words) > 0:
            content_overlap = len(orig_words & rag_words) / len(orig_words | rag_words)
        else:
            content_overlap = 0.0
        
        # Quality score (semantic similarity weighted)
        quality_score = semantic_similarity * 0.7 + content_overlap * 0.3
        
        return {
            'semantic_similarity': semantic_similarity,
            'content_overlap': content_overlap,
            'length_ratio': length_ratio,
            'quality_score': quality_score,
            'original_length': orig_len,
            'rag_length': rag_len
        }
    
    def evaluate_qa_pairs(self, evaluation_pairs: List[Dict]) -> Dict:
        """
        Evaluate Q&A pairs using Enhanced Adaptive RAG.
        
        Args:
            evaluation_pairs: List of Q&A pairs to evaluate
            
        Returns:
            Complete evaluation results with adaptive metadata
        """
        logger.info(f"🧪 Starting Enhanced Adaptive RAG evaluation on {len(evaluation_pairs)} pairs...")
        
        detailed_results = []
        failed_evaluations = 0
        
        for i, pair in enumerate(evaluation_pairs):
            try:
                question = pair.get('instruction', pair.get('question', ''))
                original_answer = pair.get('output', pair.get('answer', ''))
                
                if not question.strip() or not original_answer.strip():
                    logger.warning(f"Skipping pair {i}: empty question or answer")
                    failed_evaluations += 1
                    continue
                
                # 🚀 Enhanced Adaptive RAG Generation
                rag_answer, adaptive_metadata = self.generate_answer_with_adaptive_rag(question)
                
                # Calculate metrics
                metrics = self.calculate_evaluation_metrics(original_answer, rag_answer, question)
                
                # Store result with adaptive metadata
                result = {
                    'pair_id': i,
                    'question': question,
                    'original_answer': original_answer,
                    'rag_answer': rag_answer,
                    'metrics': metrics,
                    'adaptive_metadata': adaptive_metadata,  # 🚀 Enhanced metadata
                    'matrix_combination': pair.get('matrix_combination', '')
                }
                
                detailed_results.append(result)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(evaluation_pairs)} pairs...")
                    
            except Exception as e:
                logger.error(f"Error evaluating pair {i}: {e}")
                failed_evaluations += 1
                continue
        
        # Calculate summary statistics
        if detailed_results:
            quality_scores = [r['metrics']['quality_score'] for r in detailed_results]
            semantic_similarities = [r['metrics']['semantic_similarity'] for r in detailed_results]
            content_overlaps = [r['metrics']['content_overlap'] for r in detailed_results]
            
            # Enhanced adaptive-specific metrics
            strategy_usage = {}
            for r in detailed_results:
                strategy = r['adaptive_metadata']['strategy_used']
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            avg_context_quality = np.mean([r['adaptive_metadata']['context_quality'] for r in detailed_results])
            avg_retrieved_items = np.mean([r['adaptive_metadata']['num_retrieved'] for r in detailed_results])
            
            evaluation_summary = {
                'total_pairs_evaluated': len(detailed_results),
                'successful_evaluations': len(detailed_results),
                'failed_evaluations': failed_evaluations,
                'average_quality_score': float(np.mean(quality_scores)),
                'std_quality_score': float(np.std(quality_scores)),
                'average_semantic_similarity': float(np.mean(semantic_similarities)),
                'average_content_overlap': float(np.mean(content_overlaps)),
                'quality_score_distribution': {
                    'min': float(np.min(quality_scores)),
                    'max': float(np.max(quality_scores)),
                    'median': float(np.median(quality_scores)),
                    'q25': float(np.percentile(quality_scores, 25)),
                    'q75': float(np.percentile(quality_scores, 75))
                },
                # 🚀 Enhanced Adaptive RAG specific metrics
                'adaptive_metrics': {
                    'strategy_usage': strategy_usage,
                    'average_context_quality': float(avg_context_quality),
                    'average_retrieved_items': float(avg_retrieved_items),
                    'enhancement_active': True
                }
            }
        else:
            evaluation_summary = {
                'total_pairs_evaluated': 0,
                'successful_evaluations': 0,
                'failed_evaluations': failed_evaluations,
                'error': 'No successful evaluations completed'
            }
        
        logger.info(f"✅ Enhanced Adaptive RAG evaluation completed: {len(detailed_results)}/{len(evaluation_pairs)} successful")
        
        return {
            'evaluation_summary': evaluation_summary,
            'detailed_results': detailed_results,
            'model_info': {
                'generation_model': self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'unknown',
                'embedding_model': 'all-MiniLM-L6-v2', 
                'use_quantization': hasattr(self.model.config, 'quantization_config'),
                'device': str(self.device),
                'enhancement_pipeline': 'adaptive_rag_pipeline_v2_enhanced'  # 🚀 Mark as enhanced
            },
            'knowledge_base_info': {
                'total_qa_pairs': len(self.qa_pairs),
                'retrieval_method': 'Enhanced Adaptive RAG with 4 improvements',
                'features': [
                    'Cross-encoder re-ranking',
                    'Hybrid dense+sparse retrieval', 
                    'Dynamic context windows',
                    'Enhanced query classification'
                ]
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Enhanced Adaptive RAG Evaluator with 4 Major Improvements')
    parser.add_argument('--qa-pairs-file', type=Path, required=True,
                       help='JSON file with Q&A pairs (used as knowledge base)')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--model-name', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Hugging Face model name for generation')
    parser.add_argument('--domain-config', default='audio_equipment_domain_questions.json',
                       help='Domain configuration file for enhanced features')
    parser.add_argument('--quantize', action='store_true',
                       help='Use 4-bit quantization')
    parser.add_argument('--device', default='auto',
                       help='Device for model loading (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    print('🚀 Enhanced Adaptive RAG Evaluator')
    print('=' * 60)
    print(f'Q&A Pairs: {args.qa_pairs_file}')
    print(f'Output Dir: {args.output_dir}')
    print(f'Model: {args.model_name}')
    print(f'Domain Config: {args.domain_config}')
    print(f'Quantization: {args.quantize}')
    print(f'Device: {args.device}')
    print()
    print('🔥 Enhanced Features Active:')
    print('  ✅ Cross-encoder re-ranking')
    print('  ✅ Hybrid dense+sparse retrieval')  
    print('  ✅ Dynamic context windows')
    print('  ✅ Enhanced query classification')
    print()
    
    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Enhanced Adaptive RAG evaluator
        evaluator = EnhancedAdaptiveRAGEvaluator(
            qa_pairs_file=args.qa_pairs_file,
            model_name=args.model_name,
            domain_config_file=args.domain_config,
            use_quantization=args.quantize,
            device=args.device
        )
        
        # Load evaluation pairs (same as knowledge base for this setup)
        with open(args.qa_pairs_file, 'r', encoding='utf-8') as f:
            evaluation_pairs = json.load(f)
        
        # Run evaluation
        results = evaluator.evaluate_qa_pairs(evaluation_pairs)
        
        # Save results
        output_file = args.output_dir / 'qa_rag_evaluation_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f'\n✅ Enhanced Adaptive RAG evaluation completed!')
        print(f'📊 Results saved: {output_file}')
        
        # Display summary
        summary = results['evaluation_summary']
        print(f'\n📈 ENHANCED ADAPTIVE RAG RESULTS:')
        print('=' * 50)
        print(f"Total Pairs Evaluated: {summary.get('total_pairs_evaluated', 0)}")
        print(f"Average Quality Score: {summary.get('average_quality_score', 0):.3f}")
        print(f"Average Similarity: {summary.get('average_semantic_similarity', 0):.3f}")
        print(f"Average Context Overlap: {summary.get('average_content_overlap', 0):.3f}")
        
        if 'adaptive_metrics' in summary:
            adaptive = summary['adaptive_metrics']
            print(f"\n🚀 ENHANCED FEATURES USAGE:")
            print(f"Average Context Quality: {adaptive.get('average_context_quality', 0):.3f}")
            print(f"Average Retrieved Items: {adaptive.get('average_retrieved_items', 0):.1f}")
            print(f"Strategy Usage: {adaptive.get('strategy_usage', {})}")
        
        print(f'\n🎯 Enhancement Pipeline: ACTIVE')
        
    except Exception as e:
        print(f'❌ Error in Enhanced Adaptive RAG evaluation: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())