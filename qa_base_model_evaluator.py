#!/usr/bin/env python3
"""
Base Model Evaluator: Test model performance without RAG
Evaluates the base model's ability to answer questions using only its training knowledge
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bert_score import score as bert_score
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModelEvaluator:
    """Evaluates base model performance without RAG retrieval"""
    
    def __init__(self, model_name: str, use_quantization: bool = False, device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing Base Model Evaluator")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Quantization: {use_quantization}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if requested
        model_kwargs = {}
        if use_quantization and self.device == "cuda":
            logger.info("🔧 Configuring 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load model
        logger.info("🧠 Loading language model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not use_quantization:
            self.model = self.model.to(self.device)
        
        logger.info("✅ Base model loaded successfully")
        
        # Load embedding model for similarity evaluation
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        logger.info("✅ Embedding model loaded for evaluation")
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using only the base model (no RAG context)"""
        # Create prompt for base model
        prompt = f"""You are an expert in audio equipment and electronics. Answer the following question based on your knowledge:

Question: {question}

Answer:"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer_start = full_response.find("Answer:") + len("Answer:")
        answer = full_response[answer_start:].strip()
        
        return answer
    
    def evaluate_qa_pairs(self, qa_pairs: List[Dict]) -> Dict:
        """Evaluate base model on Q&A pairs"""
        logger.info(f"🔍 Evaluating {len(qa_pairs)} Q&A pairs with base model...")
        
        results = []
        similarities = []
        bert_scores = []
        
        start_time = time.time()
        
        for i, pair in enumerate(qa_pairs):
            question = pair['question']
            expected_answer = pair['answer']
            
            logger.info(f"📝 Processing Q&A pair {i+1}/{len(qa_pairs)}")
            
            try:
                # Generate answer with base model
                generated_answer = self.generate_answer(question)
                
                # Calculate semantic similarity
                expected_emb = self.embedding_model.encode([expected_answer])
                generated_emb = self.embedding_model.encode([generated_answer])
                similarity = cosine_similarity(expected_emb, generated_emb)[0][0]
                
                # Calculate BERT score
                P, R, F1 = bert_score([generated_answer], [expected_answer], lang="en", verbose=False)
                bert_f1 = F1.item()
                
                result = {
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': generated_answer,
                    'similarity_score': float(similarity),
                    'bert_f1': float(bert_f1),
                    'chunk_id': pair.get('chunk_id', f'pair_{i}'),
                    'source': pair.get('source', 'unknown')
                }
                
                results.append(result)
                similarities.append(similarity)
                bert_scores.append(bert_f1)
                
                logger.info(f"   ✅ Similarity: {similarity:.3f}, BERT F1: {bert_f1:.3f}")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing pair {i+1}: {e}")
                continue
        
        end_time = time.time()
        
        # Calculate aggregate metrics
        avg_similarity = np.mean(similarities) if similarities else 0.0
        avg_bert_score = np.mean(bert_scores) if bert_scores else 0.0
        total_time = end_time - start_time
        
        # Count high-quality answers
        high_quality_pairs = sum(1 for s in similarities if s > 0.7)
        quality_retention_rate = high_quality_pairs / len(similarities) if similarities else 0.0
        
        evaluation_summary = {
            'total_pairs_evaluated': len(qa_pairs),
            'successful_evaluations': len(results),
            'failed_evaluations': len(qa_pairs) - len(results),
            'avg_similarity_score': avg_similarity,
            'avg_bert_score': avg_bert_score,
            'high_quality_pairs': high_quality_pairs,
            'quality_retention_rate': quality_retention_rate,
            'total_time_seconds': total_time,
            'avg_time_per_pair': total_time / len(qa_pairs) if qa_pairs else 0.0
        }
        
        logger.info("📊 Base Model Evaluation Complete:")
        logger.info(f"   Average Similarity: {avg_similarity:.3f}")
        logger.info(f"   Average BERT F1: {avg_bert_score:.3f}")
        logger.info(f"   High Quality Pairs: {high_quality_pairs}/{len(qa_pairs)}")
        logger.info(f"   Quality Retention: {quality_retention_rate:.3f}")
        logger.info(f"   Total Time: {total_time:.1f}s")
        
        return {
            'evaluation_summary': evaluation_summary,
            'detailed_results': results,
            'model_info': {
                'model_name': self.model_name,
                'device': self.device,
                'evaluation_type': 'base_model_no_rag'
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate base model performance without RAG")
    parser.add_argument('--qa-pairs-file', type=str, required=True,
                        help='Path to Q&A pairs JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--model-name', type=str, 
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Hugging Face model name')
    parser.add_argument('--quantization', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load Q&A pairs
    logger.info(f"Loading Q&A pairs from: {args.qa_pairs_file}")
    with open(args.qa_pairs_file, 'r') as f:
        qa_pairs = json.load(f)
    
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = BaseModelEvaluator(
        model_name=args.model_name,
        use_quantization=args.quantization,
        device=args.device
    )
    
    # Run evaluation
    logger.info("🚀 Starting base model evaluation...")
    results = evaluator.evaluate_qa_pairs(qa_pairs)
    
    # Save results
    output_file = output_dir / "qa_base_model_evaluation_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Base model evaluation complete. Results saved to: {output_file}")
    
    # Print summary
    summary = results['evaluation_summary']
    print(f"\n🎯 BASE MODEL EVALUATION SUMMARY")
    print(f"================================")
    print(f"Total Q&A Pairs: {summary['total_pairs_evaluated']}")
    print(f"Successful Evaluations: {summary['successful_evaluations']}")
    print(f"Average Similarity Score: {summary['avg_similarity_score']:.3f}")
    print(f"Average BERT F1 Score: {summary['avg_bert_score']:.3f}")
    print(f"High Quality Pairs: {summary['high_quality_pairs']}")
    print(f"Quality Retention Rate: {summary['quality_retention_rate']:.3f}")
    print(f"Total Time: {summary['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()