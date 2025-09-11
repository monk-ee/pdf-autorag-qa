#!/usr/bin/env python3
"""
RAG Demonstration with Amplifier Questions
Interactive demonstration of RAG capabilities with real-world amplifier questions
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGDemonstrator:
    """Demonstrates RAG capabilities with real amplifier questions"""
    
    def __init__(self, faiss_index_path: str, metadata_path: str, model_name: str, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", use_quantization: bool = False):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        logger.info(f"🎸 Initializing RAG Demonstration System")
        logger.info(f"   Language Model: {model_name}")
        logger.info(f"   Embedding Model: {embedding_model_name}")
        logger.info(f"   FAISS Index: {faiss_index_path}")
        
        # Load FAISS index
        logger.info("📚 Loading FAISS vector index...")
        self.faiss_index = faiss.read_index(faiss_index_path)
        logger.info(f"   Index size: {self.faiss_index.ntotal} vectors")
        
        # Load metadata
        logger.info("📋 Loading Q&A metadata...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"   Loaded metadata for {len(self.metadata)} Q&A pairs")
        
        # Load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🧠 Loading embedding model on {device}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        # Load language model
        logger.info("🤖 Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading
        model_kwargs = {}
        if use_quantization and device == "cuda":
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
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not use_quantization:
            self.model = self.model.to(device)
        
        self.device = device
        logger.info("✅ RAG system initialized successfully")
    
    def retrieve_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context for a question"""
        # Encode question
        question_embedding = self.embedding_model.encode([question])
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(question_embedding.astype(np.float32), top_k)
        
        # Get relevant Q&A pairs
        context_pairs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                pair = self.metadata[idx].copy()
                pair['retrieval_score'] = float(score)
                context_pairs.append(pair)
        
        return context_pairs
    
    def generate_rag_answer(self, question: str, context_pairs: List[Dict]) -> str:
        """Generate answer using RAG with retrieved context"""
        # Build context string
        context_text = "\n\n".join([
            f"Q: {pair.get('question', pair.get('instruction', ''))}\nA: {pair.get('answer', pair.get('output', ''))}"
            for pair in context_pairs
        ])
        
        # Create RAG prompt
        prompt = f"""Based on the following context about audio equipment and amplifiers, answer the question accurately. If the context doesn't contain enough information, say so.

Context:
{context_text}

Question: {question}

Answer:"""
        
        # Generate response
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
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
        
        # Extract answer
        answer_start = full_response.find("Answer:") + len("Answer:")
        answer = full_response[answer_start:].strip()
        
        return answer
    
    def generate_base_answer(self, question: str) -> str:
        """Generate answer using base model only (no RAG)"""
        prompt = f"""You are an expert in audio equipment and electronics. Answer the following question based on your knowledge:

Question: {question}

Answer:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_start = full_response.find("Answer:") + len("Answer:")
        answer = full_response[answer_start:].strip()
        
        return answer
    
    def demonstrate_question(self, question: str) -> Dict:
        """Demonstrate RAG vs Base model for a single question"""
        logger.info(f"🔍 Demonstrating: {question}")
        
        start_time = time.time()
        
        # Retrieve context
        retrieval_start = time.time()
        context_pairs = self.retrieve_context(question, top_k=5)
        retrieval_time = time.time() - retrieval_start
        
        # Generate RAG answer
        rag_start = time.time()
        rag_answer = self.generate_rag_answer(question, context_pairs)
        rag_time = time.time() - rag_start
        
        # Generate base model answer
        base_start = time.time()
        base_answer = self.generate_base_answer(question)
        base_time = time.time() - base_start
        
        total_time = time.time() - start_time
        
        # Calculate context relevance (similarity between question and retrieved context)
        question_emb = self.embedding_model.encode([question])
        context_relevance_scores = []
        
        for pair in context_pairs:
            question_text = pair.get('question', pair.get('instruction', ''))
            context_emb = self.embedding_model.encode([question_text])
            relevance = cosine_similarity(question_emb, context_emb)[0][0]
            context_relevance_scores.append(relevance)
        
        avg_context_relevance = np.mean(context_relevance_scores) if context_relevance_scores else 0.0
        
        result = {
            'question': question,
            'rag_answer': rag_answer,
            'base_answer': base_answer,
            'retrieved_context': [
                {
                    'question': pair.get('question', pair.get('instruction', '')),
                    'answer': pair.get('answer', pair.get('output', '')),
                    'retrieval_score': pair['retrieval_score'],
                    'source': pair.get('source', 'unknown')
                }
                for pair in context_pairs
            ],
            'metrics': {
                'retrieval_time_ms': retrieval_time * 1000,
                'rag_generation_time_ms': rag_time * 1000,
                'base_generation_time_ms': base_time * 1000,
                'total_time_ms': total_time * 1000,
                'avg_context_relevance': avg_context_relevance,
                'num_context_pairs': len(context_pairs)
            }
        }
        
        logger.info(f"   ✅ RAG answer generated in {rag_time:.2f}s")
        logger.info(f"   ✅ Base answer generated in {base_time:.2f}s")
        logger.info(f"   📊 Context relevance: {avg_context_relevance:.3f}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="RAG Demonstration with Amplifier Questions")
    parser.add_argument('--qa-faiss-index', type=str, required=True,
                        help='Path to FAISS index file')
    parser.add_argument('--qa-metadata', type=str, required=True,
                        help='Path to Q&A metadata JSON file')
    parser.add_argument('--model-name', type=str, 
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Hugging Face model name')
    parser.add_argument('--embedding-model', type=str, 
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer embedding model')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSON file for demonstration results')
    parser.add_argument('--quantization', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Demo questions about amplifiers
    demo_questions = [
        "What is the output impedance of the UAFX Ruby 63 amplifier?",
        "How do you connect the amplifier to a speaker cabinet?",
        "What is the maximum power output of this amplifier?",
        "How do I adjust the gain settings on the amplifier?",
        "What type of tubes does the amplifier use?",
        "How do I troubleshoot if the amplifier has no sound output?",
        "What are the dimensions and weight of the amplifier?",
        "Can I use this amplifier with different impedance speakers?",
        "What is the frequency response range of the amplifier?",
        "How do I properly ground the amplifier for safety?"
    ]
    
    # Initialize RAG demonstrator
    logger.info("🚀 Starting RAG Demonstration...")
    demonstrator = RAGDemonstrator(
        faiss_index_path=args.qa_faiss_index,
        metadata_path=args.qa_metadata,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model,
        use_quantization=args.quantization
    )
    
    # Run demonstrations
    demonstrations = []
    for i, question in enumerate(demo_questions):
        logger.info(f"\n📝 Demo {i+1}/{len(demo_questions)}")
        try:
            result = demonstrator.demonstrate_question(question)
            demonstrations.append(result)
        except Exception as e:
            logger.error(f"❌ Error with question {i+1}: {e}")
            continue
    
    # Calculate summary statistics
    total_questions = len(demonstrations)
    avg_retrieval_time = np.mean([d['metrics']['retrieval_time_ms'] for d in demonstrations])
    avg_rag_time = np.mean([d['metrics']['rag_generation_time_ms'] for d in demonstrations])
    avg_base_time = np.mean([d['metrics']['base_generation_time_ms'] for d in demonstrations])
    avg_context_relevance = np.mean([d['metrics']['avg_context_relevance'] for d in demonstrations])
    
    # Create comprehensive report
    report = {
        'demonstration_summary': {
            'total_questions': total_questions,
            'successful_demonstrations': len(demonstrations),
            'model_info': {
                'language_model': args.model_name,
                'embedding_model': args.embedding_model,
                'quantization_used': args.quantization
            },
            'performance_metrics': {
                'avg_retrieval_time_ms': avg_retrieval_time,
                'avg_rag_generation_time_ms': avg_rag_time,
                'avg_base_generation_time_ms': avg_base_time,
                'avg_context_relevance': avg_context_relevance,
                'rag_vs_base_speedup': (avg_base_time / avg_rag_time) if avg_rag_time > 0 else 1.0
            }
        },
        'demonstrations': demonstrations,
        'sample_comparisons': demonstrations[:3]  # First 3 for quick review
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ RAG demonstration complete. Results saved to: {output_path}")
    
    # Print summary
    print(f"\n🎸 RAG DEMONSTRATION SUMMARY")
    print(f"============================")
    print(f"Questions Demonstrated: {total_questions}")
    print(f"Average Retrieval Time: {avg_retrieval_time:.1f}ms")
    print(f"Average RAG Generation: {avg_rag_time:.1f}ms")
    print(f"Average Base Generation: {avg_base_time:.1f}ms")
    print(f"Average Context Relevance: {avg_context_relevance:.3f}")
    print(f"\n🔍 Sample Question Comparison:")
    
    if demonstrations:
        sample = demonstrations[0]
        print(f"\n❓ Question: {sample['question']}")
        print(f"\n🤖 RAG Answer: {sample['rag_answer'][:200]}...")
        print(f"\n🧠 Base Answer: {sample['base_answer'][:200]}...")
        print(f"\n📚 Retrieved Context Sources: {len(sample['retrieved_context'])}")


if __name__ == "__main__":
    main()