#!/usr/bin/env python3
"""
Q&A AutoRAG Evaluator - CORRECTED VERSION

Evaluates how well our curated Q&A knowledge base performs with RAG retrieval.
This tests whether we can retrieve the correct answers from our selected Q&A pairs.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Handle FAISS imports with graceful fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu faiss-gpu")


class QARAGEvaluator:
    """RAG-based Q&A evaluation using curated Q&A pairs as knowledge base."""
    
    def __init__(self, 
                 qa_faiss_index_path: Path,
                 qa_metadata_path: Path,
                 model_name: str,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 use_quantization: bool = False,
                 device: str = 'auto'):
        """
        Initialize Q&A RAG evaluator.
        
        Args:
            qa_faiss_index_path: Path to Q&A FAISS index file
            qa_metadata_path: Path to Q&A metadata JSON
            model_name: Hugging Face model name for generation
            embedding_model_name: Sentence transformer model name
            use_quantization: Whether to use 4-bit quantization
            device: Device for model loading ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.use_quantization = use_quantization
        
        # FORCE GPU on dedicated GPU instance
        if device == 'auto':
            self.device = 'cuda'  # Force CUDA - no CPU fallbacks!
            print(f'🚀 FORCING GPU device: {self.device} (dedicated GPU instance)')
        else:
            self.device = device
            print(f'🔧 Using specified device: {self.device}')
        
        # Load components
        self._load_qa_knowledge_base(qa_faiss_index_path, qa_metadata_path)
        self._load_embedding_model()
        self._load_generation_model()
    
    def _load_qa_knowledge_base(self, index_path: Path, metadata_path: Path):
        """Load Q&A knowledge base and FAISS index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
            
        print(f'📥 Loading Q&A FAISS index: {index_path}')
        cpu_index = faiss.read_index(str(index_path))
        
        # FORCE GPU FAISS - we're on a GPU instance, no fallbacks!
        if self.device == 'cuda':
            print('🚀 FORCING FAISS index to GPU - no fallbacks!')
            res = faiss.StandardGpuResources()
            self.qa_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print('✅ FAISS index FORCED to GPU')
        else:
            print('Using CPU FAISS index')
            self.qa_index = cpu_index
        
        print(f'📥 Loading Q&A metadata: {metadata_path}')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.qa_metadata = json.load(f)
            
        print(f'✅ Loaded knowledge base with {len(self.qa_metadata)} Q&A pairs')
    
    def _load_embedding_model(self):
        """Load sentence embedding model - FORCE GPU."""
        print(f'📥 Loading embedding model: {self.embedding_model_name}')
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        if self.device == 'cuda':
            self.embedding_model.to(self.device)  # Force embedding model to GPU
            print('✅ Embedding model loaded and FORCED to GPU')
        else:
            print('✅ Embedding model loaded')
    
    def _load_generation_model(self):
        """Load text generation model."""
        print(f'📥 Loading generation model: {self.model_name}')
        
        # Configure quantization if requested
        if self.use_quantization:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print(f'🔧 Using 4-bit quantization (device: {self.device})')
            except ImportError:
                print('⚠️ BitsAndBytesConfig not available, loading without quantization')
                quantization_config = None
                self.use_quantization = False
        else:
            quantization_config = None
            print('🔧 Loading model without quantization')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            'device_map': 'auto' if self.device == 'cuda' else None,
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if not quantization_config and self.device != 'auto':
            self.model = self.model.to(self.device)
        
        print('✅ Generation model loaded')
    
    def retrieve_relevant_qa_pairs(self, question: str, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve most relevant Q&A pairs from knowledge base.
        
        Args:
            question: Query question
            k: Number of Q&A pairs to retrieve
            
        Returns:
            Tuple of (retrieved_qa_pairs, similarity_scores)
        """
        # Embed the question
        question_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(question_embedding)
        
        # Search FAISS index
        similarities, indices = self.qa_index.search(question_embedding.astype(np.float32), k)
        
        # Get retrieved Q&A pairs
        retrieved_pairs = []
        scores = []
        
        for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
            if idx >= 0 and idx < len(self.qa_metadata):
                retrieved_pairs.append(self.qa_metadata[idx])
                scores.append(float(score))
        
        return retrieved_pairs, scores
    
    def generate_rag_answer(self, question: str, retrieved_qa_pairs: List[Dict]) -> str:
        """
        Generate answer using retrieved Q&A pairs as context.
        
        Args:
            question: Input question
            retrieved_qa_pairs: Retrieved Q&A pairs for context
            
        Returns:
            Generated answer
        """
        # Build context from retrieved Q&A pairs
        context_parts = []
        for pair in retrieved_qa_pairs:
            context_parts.append(f"Q: {pair['question']}\nA: {pair['answer']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following Q&A examples, answer the question accurately and concisely.

Context:
{context}

Question: {question}
Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = inputs.to(self.device)
        
        # Generate
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
        
        return answer
    
    def calculate_evaluation_metrics(self, original_answer: str, rag_answer: str, 
                                   question: str = "") -> Dict[str, float]:
        """
        Calculate evaluation metrics between original and RAG answers.
        
        Args:
            original_answer: Ground truth answer
            rag_answer: RAG-generated answer
            question: Original question (for context)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Length-based metrics
        orig_len = len(original_answer.split())
        rag_len = len(rag_answer.split())
        length_ratio = rag_len / orig_len if orig_len > 0 else 0
        
        # Semantic similarity using embeddings
        embeddings = self.embedding_model.encode([original_answer, rag_answer])
        semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Simple content overlap (word-level Jaccard)
        orig_words = set(original_answer.lower().split())
        rag_words = set(rag_answer.lower().split())
        
        if len(orig_words | rag_words) > 0:
            content_overlap = len(orig_words & rag_words) / len(orig_words | rag_words)
        else:
            content_overlap = 0.0
        
        # Quality score (composite metric)
        quality_score = (semantic_similarity * 0.6 + 
                        content_overlap * 0.3 + 
                        min(length_ratio, 2.0) * 0.1)  # Cap length benefit
        
        return {
            'semantic_similarity': float(semantic_similarity),
            'content_overlap': float(content_overlap),
            'length_ratio': float(length_ratio),
            'quality_score': float(quality_score),
            'original_length': orig_len,
            'rag_length': rag_len
        }
    
    def evaluate_qa_pairs(self, test_qa_pairs: List[Dict]) -> List[Dict]:
        """
        Evaluate Q&A pairs using RAG retrieval.
        
        Args:
            test_qa_pairs: List of Q&A pairs to test
            
        Returns:
            List of evaluation results
        """
        evaluation_results = []
        
        print(f'🔍 Running RAG evaluation on {len(test_qa_pairs)} Q&A pairs...')
        
        for i, pair in enumerate(test_qa_pairs):
            # Handle both HF format (instruction/output) and QA format (question/answer)
            question = pair.get('instruction', pair.get('question', ''))
            original_answer = pair.get('output', pair.get('answer', ''))
            
            try:
                print(f'🔄 Evaluating pair {i+1}/{len(test_qa_pairs)}: {question[:50]}...')
                
                # Retrieve relevant Q&A pairs from knowledge base
                retrieved_pairs, retrieval_scores = self.retrieve_relevant_qa_pairs(question, k=3)
                
                # Generate RAG answer
                rag_answer = self.generate_rag_answer(question, retrieved_pairs)
                
                # Calculate evaluation metrics
                metrics = self.calculate_evaluation_metrics(original_answer, rag_answer, question)
                
                result = {
                    'pair_id': i,
                    'question': question,
                    'original_answer': original_answer,
                    'rag_answer': rag_answer,
                    'retrieved_pairs': [
                        {
                            'question': rp['question'],
                            'answer': rp['answer'][:100] + '...' if len(rp['answer']) > 100 else rp['answer'],
                            'score': score
                        }
                        for rp, score in zip(retrieved_pairs, retrieval_scores)
                    ],
                    'metrics': metrics,
                    'source_info': pair.get('source_info', {}),
                    'matrix_combination': pair.get('matrix_combination', '')
                }
                
                evaluation_results.append(result)
                
                print(f'✅ Quality Score: {metrics["quality_score"]:.3f}, Similarity: {metrics["semantic_similarity"]:.3f}')
                
            except Exception as e:
                print(f'❌ Error evaluating pair {i}: {e}')
                # Add failed result
                evaluation_results.append({
                    'pair_id': i,
                    'question': question if 'question' in locals() else 'N/A',
                    'original_answer': original_answer if 'original_answer' in locals() else 'N/A',
                    'rag_answer': '',
                    'error': str(e),
                    'metrics': {'quality_score': 0.0, 'semantic_similarity': 0.0},
                    'source_info': pair.get('source_info', {}),
                    'matrix_combination': pair.get('matrix_combination', '')
                })
        
        return evaluation_results
    
    def generate_evaluation_report(self, evaluation_results: List[Dict], output_dir: Path):
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        valid_results = [r for r in evaluation_results if 'error' not in r]
        
        if not valid_results:
            print('❌ No valid evaluation results to analyze')
            return
        
        quality_scores = [r['metrics']['quality_score'] for r in valid_results]
        semantic_similarities = [r['metrics']['semantic_similarity'] for r in valid_results]
        content_overlaps = [r['metrics']['content_overlap'] for r in valid_results]
        
        summary_stats = {
            'total_pairs_evaluated': len(evaluation_results),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(evaluation_results) - len(valid_results),
            'average_quality_score': np.mean(quality_scores),
            'std_quality_score': np.std(quality_scores),
            'average_semantic_similarity': np.mean(semantic_similarities),
            'average_content_overlap': np.mean(content_overlaps),
            'quality_score_distribution': {
                'min': np.min(quality_scores),
                'max': np.max(quality_scores),
                'median': np.median(quality_scores),
                'q25': np.percentile(quality_scores, 25),
                'q75': np.percentile(quality_scores, 75)
            }
        }
        
        # Generate report
        report = {
            'evaluation_summary': summary_stats,
            'detailed_results': evaluation_results,
            'model_info': {
                'generation_model': self.model_name,
                'embedding_model': self.embedding_model_name,
                'use_quantization': self.use_quantization,
                'device': self.device
            },
            'knowledge_base_info': {
                'total_qa_pairs': len(self.qa_metadata),
                'retrieval_method': 'Q&A-based FAISS with cosine similarity'
            }
        }
        
        # Save detailed report
        report_path = output_dir / 'qa_rag_evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f'📊 Evaluation Report Generated: {report_path}')
        print(f'   - Average Quality Score: {summary_stats["average_quality_score"]:.3f}')
        print(f'   - Average Semantic Similarity: {summary_stats["average_semantic_similarity"]:.3f}')
        print(f'   - Successful Evaluations: {summary_stats["successful_evaluations"]}/{summary_stats["total_pairs_evaluated"]}')


def main():
    parser = argparse.ArgumentParser(description='Q&A AutoRAG Evaluator')
    parser.add_argument('--qa-pairs-file', type=Path, required=True,
                       help='JSON file with Q&A pairs to evaluate')
    parser.add_argument('--qa-faiss-index', type=Path, required=True,
                       help='Q&A FAISS index file')
    parser.add_argument('--qa-metadata', type=Path, required=True,
                       help='Q&A metadata JSON file')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--model-name', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Hugging Face model name')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model name')
    parser.add_argument('--quantization', action='store_true', default=False,
                       help='Enable 4-bit quantization for memory efficiency')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load test Q&A pairs
        print(f'📥 Loading Q&A pairs from: {args.qa_pairs_file}')
        with open(args.qa_pairs_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        print(f'📊 Loaded {len(qa_pairs)} Q&A pairs for evaluation')
        
        # Initialize evaluator
        print('🔧 Initializing Q&A RAG evaluator...')
        evaluator = QARAGEvaluator(
            qa_faiss_index_path=args.qa_faiss_index,
            qa_metadata_path=args.qa_metadata,
            model_name=args.model_name,
            embedding_model_name=args.embedding_model,
            use_quantization=args.quantization
        )
        
        print('✅ Q&A RAG evaluator initialized')
        
        # Run evaluation
        print('🔍 Running Q&A RAG evaluation...')
        evaluation_results = evaluator.evaluate_qa_pairs(qa_pairs)
        
        # Generate report
        evaluator.generate_evaluation_report(evaluation_results, args.output_dir)
        
        print('✅ Q&A RAG evaluation completed successfully!')
        
    except Exception as e:
        print(f'❌ Error during Q&A RAG evaluation: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())