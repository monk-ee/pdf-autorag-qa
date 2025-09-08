#!/usr/bin/env python3
"""
Domain Specificity Evaluation: Base Llama3:8b vs RAG-Enhanced (GPU Version)
Evaluates accuracy and quality on domain-specific questions using configurable JSON
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import BERTScorer
from tqdm import tqdm
import re
from typing import List, Dict, Tuple
import logging
import sys
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger = logging.getLogger(__name__)
    logger.warning("rank-bm25 not available, hybrid retrieval disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DomainConfig:
    """Load and manage domain-specific configuration from JSON"""
    
    def __init__(self, config_file: str = "audio_equipment_domain_questions.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load domain configuration from JSON file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Domain configuration file not found: {self.config_file}")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        logger.info(f"Loaded domain configuration: {config['domain_info']['domain']} v{config['domain_info']['version']}")
        return config
        
    @property
    def domain_terms(self) -> List[str]:
        return self.config.get('domain_terms', [])
        
    @property
    def evaluation_questions(self) -> List[Dict]:
        return self.config.get('evaluation_questions', [])
        
    @property
    def uncertainty_phrases(self) -> List[str]:
        return self.config.get('uncertainty_phrases', [])
        
    @property
    def context_templates(self) -> Dict:
        return self.config.get('context_templates', {})
        
    @property
    def domain_info(self) -> Dict:
        return self.config.get('domain_info', {})


class HybridRetriever:
    """Hybrid retrieval combining dense semantic search with sparse keyword matching"""
    
    def __init__(self, embedder, qa_texts, qa_data):
        self.embedder = embedder
        self.qa_texts = qa_texts
        self.qa_data = qa_data
        
        # Dense retrieval (semantic)
        logger.info("Building dense embeddings for hybrid retriever...")
        self.dense_embeddings = embedder.encode(qa_texts, show_progress_bar=True)
        faiss.normalize_L2(self.dense_embeddings)
        
        # Sparse retrieval (BM25 keyword matching)
        logger.info("Building BM25 index...")
        tokenized_texts = [text.lower().split() for text in qa_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        logger.info(f"Hybrid retriever initialized with {len(qa_texts)} documents")
    
    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0.0 = all sparse, 1.0 = all dense)
        """
        # Dense retrieval scores
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        dense_scores = cosine_similarity(query_embedding, self.dense_embeddings)[0]
        
        # Sparse retrieval scores
        query_tokens = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(query_tokens))
        
        # Normalize sparse scores to [0,1]
        if sparse_scores.max() > 0:
            sparse_scores = sparse_scores / sparse_scores.max()
        
        # Combine scores
        combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.qa_data):
                result = self.qa_data[idx].copy()
                result['relevance_score'] = float(combined_scores[idx])
                result['dense_score'] = float(dense_scores[idx])
                result['sparse_score'] = float(sparse_scores[idx])
                results.append(result)
        
        return results


class ContextTemplates:
    """Template-based context generation using configurable templates"""
    
    def __init__(self, domain_config: DomainConfig):
        self.templates = domain_config.context_templates
        self.domain = domain_config.domain_info.get('domain', 'unknown')
    
    def get_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level from score"""
        if confidence_score < 0.4:
            return 'low'
        elif confidence_score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def get_comparison_template(self, pairs: List[Dict], question: str, confidence_score: float = 0.0) -> str:
        template = self.templates.get('comparison', {})
        prefix = template.get('prefix', f"{self.domain.upper()} COMPARISON ANALYSIS\n\nReference Examples:\n")
        suffix = template.get('suffix', "Based on the examples above, please compare and contrast: {question}\n\n")
        
        context = prefix
        for i, pair in enumerate(pairs, 1):
            context += f"{i}. Q: {pair['instruction']}\n   A: {pair['output']}\n\n"
        
        context += suffix.format(question=question)
        
        # Add confidence-based instructions
        confidence_level = self.get_confidence_level(confidence_score)
        confidence_instructions = template.get('confidence_instructions', {})
        if confidence_level in confidence_instructions:
            context += confidence_instructions[confidence_level] + "\n"
            
        return context
    
    def get_technical_template(self, pairs: List[Dict], question: str, confidence_score: float = 0.0) -> str:
        template = self.templates.get('technical', {})
        prefix = template.get('prefix', f"{self.domain.upper()} TECHNICAL REFERENCE\n\n")
        suffix = template.get('suffix', "Technical Question: {question}\n")
        
        context = prefix
        for i, pair in enumerate(pairs, 1):
            context += f"Reference {i}:\n"
            context += f"  Question: {pair['instruction']}\n"
            context += f"  Answer: {pair['output']}\n"
            context += f"  Difficulty: {pair.get('difficulty', 'N/A')}\n\n"
        
        context += suffix.format(question=question)
        
        # Add confidence-based instructions
        confidence_level = self.get_confidence_level(confidence_score)
        confidence_instructions = template.get('confidence_instructions', {})
        if confidence_level in confidence_instructions:
            context += confidence_instructions[confidence_level] + "\n"
            
        return context
    
    def get_general_template(self, pairs: List[Dict], question: str, confidence_score: float = 0.0) -> str:
        """General template with confidence gating for any question type"""
        template = self.templates.get('general', {})
        prefix = template.get('prefix', f"{self.domain.upper()} REFERENCE INFORMATION\n\n")
        suffix = template.get('suffix', "Question: {question}\n\n")
        
        context = prefix
        for i, pair in enumerate(pairs, 1):
            context += f"Example {i}:\n"
            context += f"  Q: {pair['instruction']}\n"
            context += f"  A: {pair['output']}\n\n"
        
        context += suffix.format(question=question)
        
        # Add confidence-based instructions
        confidence_level = self.get_confidence_level(confidence_score)
        confidence_instructions = template.get('confidence_instructions', {})
        if confidence_level in confidence_instructions:
            context += confidence_instructions[confidence_level] + "\n"
            
        return context


class DomainEvaluatorGPU:
    """GPU-accelerated evaluator for domain specificity and accuracy"""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 results_dir: str = "pdf-qa-generation-results",
                 config_file: str = "audio_equipment_domain_questions.json",
                 enable_bert_score: bool = True,
                 use_quantization: bool = False):
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.enable_bert_score = enable_bert_score
        self.use_quantization = use_quantization
        
        # Load domain configuration
        self.domain_config = DomainConfig(config_file)
        self.context_templates = ContextTemplates(self.domain_config)
        
        # Initialize models
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model loading based on quantization preference
        if use_quantization:
            logger.info("Using 8-bit quantization to save GPU memory")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            logger.info("Loading model without quantization (full precision)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use better embedding model for improved retrieval quality
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        logger.info(f"Using embedding model: all-mpnet-base-v2 (768-dim, better semantic understanding)")
        
        if enable_bert_score:
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device='cuda')
        else:
            self.bert_scorer = None
            
        self.qa_data = []
        self.faiss_index = None
        self.qa_texts = []
        self.hybrid_retriever = None
        
    def load_qa_data(self) -> List[Dict]:
        """Load all Q&A pairs from JSONL files"""
        logger.info("Loading Q&A data from results...")
        
        jsonl_files = list(self.results_dir.glob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        all_qa = []
        for file_path in jsonl_files:
            # Extract metadata from filename
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 4:
                difficulty = parts[2]  # basic/intermediate/advanced
                style = parts[3]       # high, balanced, conservative
            else:
                difficulty = "unknown"
                style = "unknown"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        qa_pair = json.loads(line.strip())
                        qa_pair['source_file'] = filename
                        qa_pair['difficulty'] = difficulty
                        qa_pair['style'] = style
                        qa_pair['line_num'] = line_num
                        all_qa.append(qa_pair)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON in {file_path}:{line_num} - {e}")
        
        logger.info(f"Loaded {len(all_qa)} Q&A pairs total")
        self.qa_data = all_qa
        return all_qa
    
    def build_faiss_index(self):
        """Build FAISS GPU index for Q&A retrieval"""
        logger.info("Building FAISS GPU index for Q&A retrieval...")
        
        if not self.qa_data:
            self.load_qa_data()
        
        # Create text representations for embedding
        self.qa_texts = []
        for qa in self.qa_data:
            # Combine question and answer for richer context
            text = f"Q: {qa['instruction']} A: {qa['output']}"
            self.qa_texts.append(text)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(self.qa_texts, show_progress_bar=True)
        
        # Build FAISS GPU index
        dimension = embeddings.shape[1]
        
        # Use GPU index if available
        if faiss.get_num_gpus() > 0:
            logger.info(f"Using GPU FAISS index with {faiss.get_num_gpus()} GPUs")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            logger.info("Using CPU FAISS index (no GPU available)")
            self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Initialize hybrid retriever if BM25 is available
        if HAS_BM25:
            logger.info("Initializing hybrid retriever with BM25...")
            self.hybrid_retriever = HybridRetriever(self.embedder, self.qa_texts, self.qa_data)
        else:
            logger.info("Using dense-only retrieval (BM25 not available)")
    
    def retrieve_context(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant Q&A pairs for RAG context using hybrid retrieval if available"""
        if self.faiss_index is None:
            self.build_faiss_index()
        
        # Use hybrid retrieval if available, otherwise fall back to dense-only
        if self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(question, top_k=top_k)
        else:
            # Original dense-only retrieval
            query_embedding = self.embedder.encode([question])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.qa_data):  # Valid index
                    qa_pair = self.qa_data[idx].copy()
                    qa_pair['relevance_score'] = float(score)
                    results.append(qa_pair)
            
            return results
    
    def format_smart_context(self, question: str, retrieved_pairs: List[Dict]) -> str:
        """Dynamically select context template based on question analysis with confidence gating"""
        
        question_lower = question.lower()
        
        # Calculate average confidence from relevance scores
        avg_relevance = np.mean([p.get('relevance_score', 0) for p in retrieved_pairs]) if retrieved_pairs else 0.0
        
        # Determine question type and select appropriate template with confidence
        if any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return self.context_templates.get_comparison_template(retrieved_pairs, question, avg_relevance)
        
        elif any(word in question_lower for word in ['specification', 'spec', 'technical', 'impedance', 'power', 'amplifier']):
            return self.context_templates.get_technical_template(retrieved_pairs, question, avg_relevance)
        
        elif any(word in question_lower for word in ['how to', 'steps', 'process', 'setup', 'connect']):
            # For process questions, use general template with confidence gating
            return self.context_templates.get_general_template(retrieved_pairs[:2], question, avg_relevance)
        
        else:
            # Standard template with confidence gating
            return self.context_templates.get_general_template(retrieved_pairs[:3], question, avg_relevance)
    
    def query_model(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Query the Llama model directly"""
        try:
            # Format prompt for Llama3 Instruct
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return f"ERROR: {str(e)}"
    
    def create_evaluation_questions(self) -> List[Dict]:
        """Get evaluation questions from domain configuration"""
        eval_questions = []
        
        # Load questions from configuration
        for q_config in self.domain_config.evaluation_questions:
            eval_questions.append({
                'question': q_config['question'],
                'category': q_config['category'],
                'difficulty': q_config['difficulty'],
                'expected_terms': q_config.get('expected_terms', [])
            })
        
        # Extract some real questions from the data for in-domain tests
        sample_questions = []
        if self.qa_data:
            # Get a diverse sample
            basic_qs = [qa for qa in self.qa_data if qa.get('difficulty') == 'basic'][:2]
            intermediate_qs = [qa for qa in self.qa_data if qa.get('difficulty') == 'intermediate'][:2]
            advanced_qs = [qa for qa in self.qa_data if qa.get('difficulty') == 'advanced'][:2]
            
            for qa in basic_qs + intermediate_qs + advanced_qs:
                sample_questions.append({
                    'question': qa['instruction'],
                    'ground_truth': qa['output'],
                    'category': 'in_domain_real',
                    'difficulty': qa.get('difficulty', 'unknown'),
                    'source': qa.get('source_file', 'unknown'),
                    'expected_terms': []
                })
        
        return eval_questions + sample_questions
    
    def calculate_similarity_score(self, answer1: str, answer2: str) -> float:
        """Calculate semantic similarity between two answers"""
        if not answer1.strip() or not answer2.strip():
            return 0.0
        
        embeddings = self.embedder.encode([answer1, answer2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def evaluate_answer_quality(self, question: str, answer: str, ground_truth: str = None, expected_terms: List[str] = None) -> Dict:
        """Evaluate answer quality with multiple metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['length'] = len(answer)
        metrics['word_count'] = len(answer.split())
        
        # Check for error responses
        metrics['has_error'] = 'ERROR:' in answer
        
        # Domain-specific indicators using configurable terms
        domain_terms = self.domain_config.domain_terms
        answer_lower = answer.lower()
        domain_mentions = sum(1 for term in domain_terms if term.lower() in answer_lower)
        metrics['domain_term_count'] = domain_mentions
        metrics['domain_relevance'] = min(domain_mentions / 5.0, 1.0)  # Normalize to 0-1
        
        # Expected terms coverage (if provided)
        if expected_terms:
            expected_mentions = sum(1 for term in expected_terms if term.lower() in answer_lower)
            metrics['expected_term_count'] = expected_mentions
            metrics['expected_term_coverage'] = expected_mentions / len(expected_terms) if expected_terms else 0.0
        else:
            metrics['expected_term_count'] = 0
            metrics['expected_term_coverage'] = 0.0
        
        # Uncertainty indicators using configurable phrases
        uncertainty_phrases = self.domain_config.uncertainty_phrases
        has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
        metrics['shows_uncertainty'] = has_uncertainty
        
        # If ground truth provided, calculate similarity
        if ground_truth:
            # Semantic similarity via embeddings
            metrics['ground_truth_similarity'] = self.calculate_similarity_score(answer, ground_truth)
            
            # BERT-score for better semantic evaluation
            if self.bert_scorer:
                try:
                    P, R, F1 = self.bert_scorer.score([answer], [ground_truth])
                    metrics['bert_precision'] = float(P[0])
                    metrics['bert_recall'] = float(R[0]) 
                    metrics['bert_f1'] = float(F1[0])
                except Exception as e:
                    logger.warning(f"BERT-score calculation failed: {e}")
                    metrics['bert_precision'] = 0.0
                    metrics['bert_recall'] = 0.0
                    metrics['bert_f1'] = 0.0
            else:
                metrics['bert_precision'] = 0.0
                metrics['bert_recall'] = 0.0
                metrics['bert_f1'] = 0.0
        
        return metrics
    
    def run_evaluation(self, max_questions: int = 15) -> pd.DataFrame:
        """Run the full evaluation comparing base vs RAG-enhanced model"""
        domain_name = self.domain_config.domain_info.get('domain', 'unknown')
        logger.info(f"Starting {domain_name} domain evaluation with model: {self.model_name}")
        
        # Ensure we have data and index
        if not self.qa_data:
            self.load_qa_data()
        if self.faiss_index is None:
            self.build_faiss_index()
        
        # Get evaluation questions
        eval_questions = self.create_evaluation_questions()[:max_questions]
        logger.info(f"Running evaluation on {len(eval_questions)} questions")
        
        results = []
        
        for i, q_data in enumerate(tqdm(eval_questions, desc="Evaluating")):
            question = q_data['question']
            category = q_data['category'] 
            ground_truth = q_data.get('ground_truth', None)
            expected_terms = q_data.get('expected_terms', [])
            
            logger.info(f"Question {i+1}: {question[:50]}...")
            
            # Base model response
            try:
                base_answer = self.query_model(question)
                base_metrics = self.evaluate_answer_quality(question, base_answer, ground_truth, expected_terms)
            except Exception as e:
                logger.error(f"Error with base model: {e}")
                base_answer = f"ERROR: {e}"
                base_metrics = {'has_error': True}
            
            # RAG-enhanced response with confidence gating
            try:
                retrieved = self.retrieve_context(question, top_k=3)
                # Use smart context formatting with confidence gating
                rag_context = self.format_smart_context(question, retrieved)
                
                rag_answer = self.query_model(rag_context)
                rag_metrics = self.evaluate_answer_quality(question, rag_answer, ground_truth, expected_terms)
                rag_metrics['context_pairs_used'] = len(retrieved)
                rag_metrics['avg_context_relevance'] = np.mean([p['relevance_score'] for p in retrieved]) if retrieved else 0
                
                # Add confidence score used for gating
                avg_confidence = np.mean([p.get('relevance_score', 0) for p in retrieved]) if retrieved else 0.0
                rag_metrics['confidence_score'] = avg_confidence
                
                # Add hybrid retrieval metrics if available
                if retrieved and 'dense_score' in retrieved[0]:
                    rag_metrics['avg_dense_score'] = np.mean([p['dense_score'] for p in retrieved])
                    rag_metrics['avg_sparse_score'] = np.mean([p['sparse_score'] for p in retrieved])
                    
            except Exception as e:
                logger.error(f"Error with RAG model: {e}")
                rag_answer = f"ERROR: {e}"
                rag_metrics = {'has_error': True}
                retrieved = []
            
            # Store results
            result = {
                'question_id': i,
                'question': question,
                'category': category,
                'difficulty': q_data.get('difficulty', 'unknown'),
                'expected_terms': expected_terms,
                'ground_truth': ground_truth,
                'base_answer': base_answer,
                'rag_answer': rag_answer,
                'retrieved_context': [p['instruction'][:100] + '...' for p in retrieved[:2]], # First 2 for brevity
            }
            
            # Add metrics with prefixes
            for k, v in base_metrics.items():
                result[f'base_{k}'] = v
            for k, v in rag_metrics.items():
                result[f'rag_{k}'] = v
            
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze evaluation results and generate summary metrics"""
        analysis = {}
        domain_name = self.domain_config.domain_info.get('domain', 'unknown')
        
        # System configuration
        analysis['system_config'] = {
            'domain': domain_name,
            'domain_version': self.domain_config.domain_info.get('version', 'unknown'),
            'model_name': self.model_name,
            'quantization_enabled': self.use_quantization,
            'quantization_type': '8-bit' if self.use_quantization else 'none',
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'faiss_gpu_enabled': faiss.get_num_gpus() > 0 if 'faiss' in sys.modules else False
        }
        
        # GPU memory info if available
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
            memory_total = device_props.total_memory / (1024**3)           # GB
            
            analysis['system_config'].update({
                'gpu_name': device_props.name,
                'gpu_memory_total_gb': round(memory_total, 2),
                'gpu_memory_allocated_gb': round(memory_allocated, 2),
                'gpu_memory_reserved_gb': round(memory_reserved, 2),
                'gpu_memory_utilization': round((memory_reserved / memory_total) * 100, 1)
            })
        
        # Overall metrics
        analysis['total_questions'] = len(results_df)
        analysis['categories'] = results_df['category'].value_counts().to_dict()
        
        # Error rates
        analysis['base_error_rate'] = results_df['base_has_error'].mean() if 'base_has_error' in results_df.columns else 0
        analysis['rag_error_rate'] = results_df['rag_has_error'].mean() if 'rag_has_error' in results_df.columns else 0
        
        # Domain relevance (higher is better for in-domain, lower might be better for out-domain)
        analysis['avg_base_domain_relevance'] = results_df['base_domain_relevance'].mean()
        analysis['avg_rag_domain_relevance'] = results_df['rag_domain_relevance'].mean()
        
        # Expected terms coverage
        analysis['avg_base_expected_coverage'] = results_df['base_expected_term_coverage'].mean()
        analysis['avg_rag_expected_coverage'] = results_df['rag_expected_term_coverage'].mean()
        
        # Answer lengths
        analysis['avg_base_word_count'] = results_df['base_word_count'].mean()
        analysis['avg_rag_word_count'] = results_df['rag_word_count'].mean()
        
        # Uncertainty (good for out-of-domain questions)
        analysis['base_uncertainty_rate'] = results_df['base_shows_uncertainty'].mean()
        analysis['rag_uncertainty_rate'] = results_df['rag_shows_uncertainty'].mean()
        
        # BERT scores if available
        if 'base_bert_f1' in results_df.columns:
            analysis['avg_base_bert_f1'] = results_df['base_bert_f1'].mean()
            analysis['avg_rag_bert_f1'] = results_df['rag_bert_f1'].mean()
        
        # By category analysis
        category_analysis = {}
        for category in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == category]
            category_analysis[category] = {
                'count': len(cat_df),
                'base_domain_relevance': cat_df['base_domain_relevance'].mean(),
                'rag_domain_relevance': cat_df['rag_domain_relevance'].mean(),
                'base_expected_coverage': cat_df['base_expected_term_coverage'].mean(),
                'rag_expected_coverage': cat_df['rag_expected_term_coverage'].mean(),
                'base_uncertainty': cat_df['base_shows_uncertainty'].mean(),
                'rag_uncertainty': cat_df['rag_shows_uncertainty'].mean(),
            }
            
            # Ground truth similarity if available (handle NaN properly)
            if 'base_ground_truth_similarity' in cat_df.columns:
                base_sim = cat_df['base_ground_truth_similarity'].dropna()
                rag_sim = cat_df['rag_ground_truth_similarity'].dropna()
                category_analysis[category]['base_similarity'] = base_sim.mean() if len(base_sim) > 0 else None
                category_analysis[category]['rag_similarity'] = rag_sim.mean() if len(rag_sim) > 0 else None
            
            # BERT scores if available (handle NaN properly)
            if 'base_bert_f1' in cat_df.columns:
                base_bert = cat_df['base_bert_f1'].dropna()
                rag_bert = cat_df['rag_bert_f1'].dropna()
                category_analysis[category]['base_bert_f1'] = base_bert.mean() if len(base_bert) > 0 else None
                category_analysis[category]['rag_bert_f1'] = rag_bert.mean() if len(rag_bert) > 0 else None
        
        analysis['by_category'] = category_analysis
        
        return analysis
    
    def save_detailed_responses(self, results_df: pd.DataFrame, output_file: str) -> None:
        """Save detailed responses for manual review and pipeline artifacts"""
        import datetime
        
        domain_info = self.domain_config.domain_info
        
        # Prepare detailed responses structure
        detailed_responses = {
            "evaluation_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "domain": domain_info.get('domain', 'unknown'),
                "domain_version": domain_info.get('version', 'unknown'),
                "model_name": self.model_name,
                "embedding_model": "all-mpnet-base-v2",
                "quantization_enabled": self.use_quantization,
                "total_questions": len(results_df),
                "retrieval_method": "hybrid" if self.hybrid_retriever else "dense_only",
                "has_bm25": HAS_BM25,
                "evaluation_version": "v3.0_configurable_domain"
            },
            "questions_and_responses": []
        }
        
        # Process each question and response pair
        for _, row in results_df.iterrows():
            question_data = {
                "question_id": int(row['question_id']),
                "question": row['question'],
                "category": row['category'],
                "difficulty": row.get('difficulty', 'unknown'),
                "expected_terms": row.get('expected_terms', []),
                "ground_truth": row.get('ground_truth', None),
                
                # Base model response
                "base_model": {
                    "response": row['base_answer'],
                    "metrics": {
                        "word_count": int(row.get('base_word_count', 0)),
                        "domain_relevance": float(row.get('base_domain_relevance', 0)),
                        "domain_term_count": int(row.get('base_domain_term_count', 0)),
                        "expected_term_coverage": float(row.get('base_expected_term_coverage', 0)),
                        "expected_term_count": int(row.get('base_expected_term_count', 0)),
                        "shows_uncertainty": bool(row.get('base_shows_uncertainty', False)),
                        "has_error": bool(row.get('base_has_error', False)),
                        "bert_f1": float(row.get('base_bert_f1', 0)) if pd.notna(row.get('base_bert_f1', 0)) else None,
                        "bert_precision": float(row.get('base_bert_precision', 0)) if pd.notna(row.get('base_bert_precision', 0)) else None,
                        "bert_recall": float(row.get('base_bert_recall', 0)) if pd.notna(row.get('base_bert_recall', 0)) else None,
                        "ground_truth_similarity": float(row.get('base_ground_truth_similarity', 0)) if pd.notna(row.get('base_ground_truth_similarity', 0)) else None
                    }
                },
                
                # RAG-enhanced model response
                "rag_model": {
                    "response": row['rag_answer'],
                    "context_used": {
                        "pairs_used": int(row.get('rag_context_pairs_used', 0)),
                        "avg_relevance": float(row.get('rag_avg_context_relevance', 0)),
                        "confidence_score": float(row.get('rag_confidence_score', 0)),
                        "retrieved_context_preview": row.get('retrieved_context', []),
                        "avg_dense_score": float(row.get('rag_avg_dense_score', 0)) if pd.notna(row.get('rag_avg_dense_score', 0)) else None,
                        "avg_sparse_score": float(row.get('rag_avg_sparse_score', 0)) if pd.notna(row.get('rag_avg_sparse_score', 0)) else None
                    },
                    "metrics": {
                        "word_count": int(row.get('rag_word_count', 0)),
                        "domain_relevance": float(row.get('rag_domain_relevance', 0)),
                        "domain_term_count": int(row.get('rag_domain_term_count', 0)),
                        "expected_term_coverage": float(row.get('rag_expected_term_coverage', 0)),
                        "expected_term_count": int(row.get('rag_expected_term_count', 0)),
                        "shows_uncertainty": bool(row.get('rag_shows_uncertainty', False)),
                        "has_error": bool(row.get('rag_has_error', False)),
                        "bert_f1": float(row.get('rag_bert_f1', 0)) if pd.notna(row.get('rag_bert_f1', 0)) else None,
                        "bert_precision": float(row.get('rag_bert_precision', 0)) if pd.notna(row.get('rag_bert_precision', 0)) else None,
                        "bert_recall": float(row.get('rag_bert_recall', 0)) if pd.notna(row.get('rag_bert_recall', 0)) else None,
                        "ground_truth_similarity": float(row.get('rag_ground_truth_similarity', 0)) if pd.notna(row.get('rag_ground_truth_similarity', 0)) else None
                    }
                },
                
                # Comparison metrics
                "comparison": {
                    "bert_f1_improvement": float(row.get('rag_bert_f1', 0)) - float(row.get('base_bert_f1', 0)) if pd.notna(row.get('rag_bert_f1', 0)) and pd.notna(row.get('base_bert_f1', 0)) else None,
                    "domain_relevance_change": float(row.get('rag_domain_relevance', 0)) - float(row.get('base_domain_relevance', 0)),
                    "expected_coverage_change": float(row.get('rag_expected_term_coverage', 0)) - float(row.get('base_expected_term_coverage', 0)),
                    "word_count_change": int(row.get('rag_word_count', 0)) - int(row.get('base_word_count', 0)),
                    "rag_is_better_bert": float(row.get('rag_bert_f1', 0)) > float(row.get('base_bert_f1', 0)) if pd.notna(row.get('rag_bert_f1', 0)) and pd.notna(row.get('base_bert_f1', 0)) else None,
                    "rag_is_more_domain_relevant": float(row.get('rag_domain_relevance', 0)) > float(row.get('base_domain_relevance', 0))
                }
            }
            
            detailed_responses["questions_and_responses"].append(question_data)
        
        # Add summary statistics
        detailed_responses["summary"] = {
            "avg_bert_f1_improvement": np.mean([
                q["comparison"]["bert_f1_improvement"] 
                for q in detailed_responses["questions_and_responses"] 
                if q["comparison"]["bert_f1_improvement"] is not None
            ]) if any(q["comparison"]["bert_f1_improvement"] is not None for q in detailed_responses["questions_and_responses"]) else None,
            
            "questions_where_rag_is_better": sum(
                1 for q in detailed_responses["questions_and_responses"] 
                if q["comparison"]["rag_is_better_bert"] is True
            ),
            
            "questions_where_rag_more_domain_relevant": sum(
                1 for q in detailed_responses["questions_and_responses"] 
                if q["comparison"]["rag_is_more_domain_relevant"] is True
            ),
            
            "by_category_improvements": {}
        }
        
        # Calculate category-specific improvements
        for category in results_df['category'].unique():
            category_questions = [
                q for q in detailed_responses["questions_and_responses"] 
                if q["category"] == category
            ]
            
            if category_questions:
                bert_improvements = [
                    q["comparison"]["bert_f1_improvement"] 
                    for q in category_questions 
                    if q["comparison"]["bert_f1_improvement"] is not None
                ]
                
                detailed_responses["summary"]["by_category_improvements"][category] = {
                    "count": len(category_questions),
                    "avg_bert_f1_improvement": np.mean(bert_improvements) if bert_improvements else None,
                    "questions_where_rag_better": sum(
                        1 for q in category_questions 
                        if q["comparison"]["rag_is_better_bert"] is True
                    )
                }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_responses, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved detailed responses for manual review: {output_file}")
        logger.info(f"File contains {len(detailed_responses['questions_and_responses'])} question-answer pairs")


def main():
    """Run the domain specificity evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Domain Specificity Evaluation')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name')
    parser.add_argument('--results-dir', default='pdf-qa-generation-results', help='Directory containing Q&A generation results')
    parser.add_argument('--config', default='audio_equipment_domain_questions.json', help='Domain configuration JSON file')
    parser.add_argument('--max-questions', type=int, default=15, help='Maximum questions to evaluate')
    parser.add_argument('--no-bert-score', action='store_true', help='Disable BERT score')
    parser.add_argument('--quantize', action='store_true', help='Use 8-bit quantization to save GPU memory')
    
    args = parser.parse_args()
    
    logger.info("Starting Domain Specificity Evaluation (GPU)")
    
    evaluator = DomainEvaluatorGPU(
        model_name=args.model,
        results_dir=args.results_dir,
        config_file=args.config,
        enable_bert_score=not args.no_bert_score,
        use_quantization=args.quantize
    )
    
    # Run evaluation
    results_df = evaluator.run_evaluation(max_questions=args.max_questions)
    
    # Save detailed results
    results_file = "domain_eval_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Save detailed responses for manual review
    responses_file = "domain_eval_responses.json"
    evaluator.save_detailed_responses(results_df, responses_file)
    logger.info(f"Detailed responses saved to: {responses_file}")
    
    # Analyze and save summary
    analysis = evaluator.analyze_results(results_df)
    
    analysis_file = "domain_eval_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Analysis saved to: {analysis_file}")
    
    # Print summary
    domain_name = evaluator.domain_config.domain_info.get('domain', 'unknown').upper()
    print("\n" + "="*80)
    print(f"{domain_name} DOMAIN SPECIFICITY EVALUATION SUMMARY")
    print("="*80)
    
    # System configuration
    config = analysis['system_config']
    print(f"🖥️  SYSTEM CONFIGURATION:")
    print(f"   Domain: {config['domain']} v{config['domain_version']}")
    print(f"   Model: {config['model_name']}")
    print(f"   Quantization: {config['quantization_type']}")
    print(f"   CUDA Available: {config['cuda_available']}")
    if config['cuda_available']:
        print(f"   GPU: {config.get('gpu_name', 'Unknown')} ({config['gpu_count']} device(s))")
        if 'gpu_memory_total_gb' in config:
            print(f"   GPU Memory: {config['gpu_memory_utilization']}% used ({config['gpu_memory_reserved_gb']:.1f}/{config['gpu_memory_total_gb']:.1f} GB)")
    print(f"   FAISS GPU: {'Enabled' if config['faiss_gpu_enabled'] else 'CPU-only'}")
    
    print(f"\nTotal Questions: {analysis['total_questions']}")
    print(f"Categories: {list(analysis['categories'].keys())}")
    
    print(f"\nError Rates:")
    print(f"  Base Model: {analysis['base_error_rate']:.1%}")
    print(f"  RAG Model:  {analysis['rag_error_rate']:.1%}")
    
    print(f"\nDomain Relevance (0-1, higher = more domain-specific):")
    print(f"  Base Model: {analysis['avg_base_domain_relevance']:.3f}")
    print(f"  RAG Model:  {analysis['avg_rag_domain_relevance']:.3f}")
    
    print(f"\nExpected Terms Coverage (0-1, higher = better):")
    print(f"  Base Model: {analysis['avg_base_expected_coverage']:.3f}")
    print(f"  RAG Model:  {analysis['avg_rag_expected_coverage']:.3f}")
    
    print(f"\nAnswer Lengths:")
    print(f"  Base Model: {analysis['avg_base_word_count']:.1f} words")
    print(f"  RAG Model:  {analysis['avg_rag_word_count']:.1f} words")
    
    print(f"\nUncertainty Indicators (good for out-of-domain):")
    print(f"  Base Model: {analysis['base_uncertainty_rate']:.1%}")
    print(f"  RAG Model:  {analysis['rag_uncertainty_rate']:.1%}")
    
    if 'avg_base_bert_f1' in analysis:
        print(f"\nBERT F1 Scores:")
        print(f"  Base Model: {analysis['avg_base_bert_f1']:.3f}")
        print(f"  RAG Model:  {analysis['avg_rag_bert_f1']:.3f}")
    
    print("="*80)
    logger.info("Domain evaluation complete!")


if __name__ == "__main__":
    main()