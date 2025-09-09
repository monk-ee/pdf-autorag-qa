#!/usr/bin/env python3
"""
Adaptive RAG Pipeline: Dynamic retrieval and response optimization
Adjusts retrieval strategy and context based on query analysis and confidence scores
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, replace
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis results for a query"""
    query_type: str  # factual, conceptual, procedural, technical, comparison
    domain_relevance: float  # 0-1 score
    complexity: str  # basic, intermediate, advanced
    uncertainty_required: bool
    key_concepts: List[str]
    confidence_threshold: float


@dataclass
class RetrievalStrategy:
    """Retrieval strategy configuration"""
    top_k: int
    alpha: float  # dense vs sparse weight
    rerank: bool
    context_window: int
    confidence_gating: bool
    fallback_strategy: str


@dataclass
class AdaptiveResponse:
    """Response with adaptive metadata"""
    answer: str
    confidence_scores: Dict[str, float]
    retrieval_used: RetrievalStrategy
    context_quality: float
    adaptive_adjustments: List[str]


class QueryAnalyzer:
    """Analyzes queries to determine optimal RAG strategy"""
    
    def __init__(self, domain_config_file: str = "audio_equipment_domain_questions.json"):
        self.config_file = Path(domain_config_file)
        self.config = self.load_config()
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Pre-compute category embeddings for classification
        self.category_embeddings = self._build_category_embeddings()
        
    def load_config(self) -> Dict:
        """Load domain configuration"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Domain config not found: {self.config_file}")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Build embeddings for query type classification"""
        categories = {
            "factual": ["what is", "define", "explain the definition", "meaning of"],
            "conceptual": ["how does", "why does", "what happens when", "relationship between"],
            "procedural": ["how to", "steps to", "process for", "way to", "method"],
            "technical": ["specifications", "calculate", "measure", "parameters", "ratings"],
            "comparison": ["difference between", "compare", "versus", "better than", "pros and cons"],
            "troubleshooting": ["problem with", "issue", "not working", "diagnose", "fix", "troubleshoot"]
        }
        
        embeddings = {}
        for category, examples in categories.items():
            category_text = " ".join(examples)
            embeddings[category] = self.embedder.encode([category_text])[0]
            
        return embeddings
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine optimal RAG approach"""
        query_lower = query.lower()
        
        # Query type classification
        query_embedding = self.embedder.encode([query])[0]
        
        similarities = {}
        for category, category_embedding in self.category_embeddings.items():
            sim = cosine_similarity([query_embedding], [category_embedding])[0][0]
            similarities[category] = float(sim)
        
        query_type = max(similarities, key=similarities.get)
        
        # Domain relevance scoring
        domain_terms = self.config.get('domain_terms', {})
        all_terms = []
        for category_terms in domain_terms.values():
            all_terms.extend(category_terms)
        
        domain_matches = sum(1 for term in all_terms if term.lower() in query_lower)
        domain_relevance = min(domain_matches / 5.0, 1.0)  # Normalize to 0-1
        
        # Complexity assessment
        complexity_indicators = {
            'basic': ['what is', 'define', 'simple', 'basic'],
            'intermediate': ['how does', 'why', 'compare', 'difference'],
            'advanced': ['calculate', 'optimize', 'design', 'troubleshoot', 'specifications']
        }
        
        complexity = 'basic'  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
        
        # Uncertainty requirements
        uncertainty_phrases = self.config.get('uncertainty_phrases', {}).get('appropriate_uncertainty', [])
        out_domain_indicators = ['car', 'programming', 'cooking', 'weather', 'sports']
        uncertainty_required = (
            domain_relevance < 0.3 or 
            any(indicator in query_lower for indicator in out_domain_indicators)
        )
        
        # Extract key concepts
        key_concepts = [term for term in all_terms if term.lower() in query_lower]
        
        # Set confidence threshold based on domain relevance and complexity
        if domain_relevance > 0.7 and complexity in ['basic', 'intermediate']:
            confidence_threshold = 0.7
        elif domain_relevance > 0.5:
            confidence_threshold = 0.5
        else:
            confidence_threshold = 0.3
        
        return QueryAnalysis(
            query_type=query_type,
            domain_relevance=domain_relevance,
            complexity=complexity,
            uncertainty_required=uncertainty_required,
            key_concepts=key_concepts,
            confidence_threshold=confidence_threshold
        )


class AdaptiveRetriever:
    """Adaptive retrieval system that adjusts strategy based on query analysis"""
    
    def __init__(self, qa_data: List[Dict], embedder: SentenceTransformer):
        self.qa_data = qa_data
        self.embedder = embedder
        
        # Build indices
        self._build_indices()
        
        # Strategy configurations
        self.strategies = {
            'conservative': RetrievalStrategy(
                top_k=2, alpha=0.8, rerank=False, 
                context_window=512, confidence_gating=True, 
                fallback_strategy='uncertainty'
            ),
            'balanced': RetrievalStrategy(
                top_k=3, alpha=0.7, rerank=True,
                context_window=768, confidence_gating=True,
                fallback_strategy='context_expansion'
            ),
            'aggressive': RetrievalStrategy(
                top_k=5, alpha=0.6, rerank=True,
                context_window=1024, confidence_gating=False,
                fallback_strategy='multi_strategy'
            )
        }
    
    def _build_indices(self):
        """Build FAISS and BM25 indices"""
        logger.info("Building adaptive retrieval indices...")
        
        # Create text representations
        self.qa_texts = []
        for qa in self.qa_data:
            text = f"Q: {qa['instruction']} A: {qa['output']}"
            self.qa_texts.append(text)
        
        # Dense embeddings
        logger.info("Generating embeddings for adaptive retrieval...")
        embeddings = self.embedder.encode(self.qa_texts, show_progress_bar=True)
        
        # FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        # BM25 index (if available)
        try:
            from rank_bm25 import BM25Okapi
            tokenized_texts = [text.lower().split() for text in self.qa_texts]
            self.bm25_index = BM25Okapi(tokenized_texts)
            self.has_bm25 = True
        except ImportError:
            self.bm25_index = None
            self.has_bm25 = False
            logger.warning("BM25 not available, using dense-only retrieval")
    
    def select_strategy(self, analysis: QueryAnalysis) -> str:
        """Select retrieval strategy based on query analysis"""
        if analysis.domain_relevance < 0.3:
            return 'conservative'
        elif analysis.complexity == 'advanced' or analysis.query_type == 'technical':
            return 'aggressive'
        else:
            return 'balanced'
    
    def retrieve_adaptive(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Dict], str]:
        """Perform adaptive retrieval based on query analysis"""
        strategy_name = self.select_strategy(analysis)
        strategy = self.strategies[strategy_name]
        
        # Dense retrieval
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, strategy.top_k * 2)
        
        # Sparse retrieval (if available)
        if self.has_bm25 and strategy.alpha < 1.0:
            query_tokens = query.lower().split()
            sparse_scores = np.array(self.bm25_index.get_scores(query_tokens))
            
            # Normalize sparse scores
            if sparse_scores.max() > 0:
                sparse_scores = sparse_scores / sparse_scores.max()
            
            # Combine scores for all documents
            combined_scores = np.zeros(len(self.qa_data))
            
            # Add dense scores
            for i, (score, idx) in enumerate(zip(dense_scores[0], dense_indices[0])):
                if idx < len(self.qa_data):
                    combined_scores[idx] = strategy.alpha * score
            
            # Add sparse scores
            for i, score in enumerate(sparse_scores):
                if i < len(combined_scores):
                    combined_scores[i] += (1 - strategy.alpha) * score
            
            # Get top results
            top_indices = np.argsort(combined_scores)[::-1][:strategy.top_k]
        else:
            # Dense-only retrieval
            top_indices = dense_indices[0][:strategy.top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(self.qa_data):
                result = self.qa_data[idx].copy()
                result['relevance_score'] = float(combined_scores[idx] if self.has_bm25 else dense_scores[0][list(dense_indices[0]).index(idx)])
                results.append(result)
        
        # Apply confidence gating
        if strategy.confidence_gating:
            avg_relevance = np.mean([r['relevance_score'] for r in results]) if results else 0
            if avg_relevance < analysis.confidence_threshold:
                # Apply fallback strategy
                if strategy.fallback_strategy == 'uncertainty':
                    results = results[:1]  # Use only top result with uncertainty
                elif strategy.fallback_strategy == 'context_expansion':
                    # Expand search with lower threshold
                    expanded_strategy = strategy_name if strategy_name != 'conservative' else 'balanced'
                    return self.retrieve_adaptive(query, replace(analysis, confidence_threshold=0.2))
        
        return results, strategy_name


class AdaptiveContextFormatter:
    """Formats context adaptively based on query analysis and retrieval confidence"""
    
    def __init__(self, domain_config: Dict):
        self.config = domain_config
        self.templates = domain_config.get('context_templates', {})
    
    def format_adaptive_context(self, query: str, analysis: QueryAnalysis, 
                              retrieved: List[Dict], strategy_used: str) -> str:
        """Format context adaptively based on analysis and retrieval quality"""
        
        # Calculate context quality
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in retrieved]) if retrieved else 0
        
        # Select template based on query type
        if analysis.query_type == 'comparison':
            template = self.templates.get('comparison', {})
        elif analysis.query_type in ['technical', 'troubleshooting']:
            template = self.templates.get('technical', {})
        else:
            template = self.templates.get('general', {})
        
        # Determine confidence level for instructions
        if avg_relevance >= 0.7:
            confidence_level = 'high'
        elif avg_relevance >= 0.4:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Build context
        prefix = template.get('prefix', f"{self.config.get('domain_info', {}).get('domain', 'DOMAIN').upper()} REFERENCE INFORMATION\n\n")
        suffix = template.get('suffix', "Question: {question}\n\n")
        
        context = prefix
        
        # Add retrieved information with confidence indicators
        for i, item in enumerate(retrieved, 1):
            relevance = item.get('relevance_score', 0)
            confidence_indicator = "🔴" if relevance < 0.3 else "🟡" if relevance < 0.6 else "🟢"
            
            context += f"Reference {i} {confidence_indicator}:\n"
            context += f"  Q: {item['instruction']}\n"
            context += f"  A: {item['output']}\n"
            context += f"  Relevance: {relevance:.2f}\n\n"
        
        context += suffix.format(question=query)
        
        # Add adaptive instructions
        confidence_instructions = template.get('confidence_instructions', {})
        if confidence_level in confidence_instructions:
            context += confidence_instructions[confidence_level] + "\n"
        
        # Add strategy-specific guidance
        if strategy_used == 'conservative':
            context += "\nNOTE: Conservative retrieval used - express uncertainty if information is insufficient.\n"
        elif strategy_used == 'aggressive':
            context += f"\nNOTE: Comprehensive search performed ({len(retrieved)} references) - provide detailed analysis.\n"
        
        # Add domain boundary guidance
        if analysis.domain_relevance < 0.5:
            context += "\nIMPORTANT: This question may be outside the domain scope. Express appropriate uncertainty.\n"
        
        return context


class AdaptiveRAGPipeline:
    """Complete adaptive RAG pipeline"""
    
    def __init__(self, qa_data: List[Dict], domain_config_file: str = "audio_equipment_domain_questions.json"):
        self.qa_data = qa_data
        
        # Initialize components
        self.analyzer = QueryAnalyzer(domain_config_file)
        self.retriever = AdaptiveRetriever(qa_data, self.analyzer.embedder)
        self.formatter = AdaptiveContextFormatter(self.analyzer.config)
        
        logger.info(f"Adaptive RAG Pipeline initialized with {len(qa_data)} Q&A pairs")
    
    def process_query(self, query: str) -> Dict:
        """Process query through adaptive RAG pipeline"""
        
        # 1. Query Analysis
        analysis = self.analyzer.analyze_query(query)
        logger.info(f"Query analysis: {analysis.query_type} | Domain: {analysis.domain_relevance:.2f} | {analysis.complexity}")
        
        # 2. Adaptive Retrieval
        retrieved, strategy_used = self.retriever.retrieve_adaptive(query, analysis)
        logger.info(f"Retrieved {len(retrieved)} items using {strategy_used} strategy")
        
        # 3. Context Formatting
        formatted_context = self.formatter.format_adaptive_context(query, analysis, retrieved, strategy_used)
        
        # 4. Calculate context quality metrics
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in retrieved]) if retrieved else 0
        context_diversity = len(set(r.get('difficulty', 'unknown') for r in retrieved)) if retrieved else 0
        
        # 5. Prepare adaptive metadata
        adaptive_adjustments = []
        if strategy_used == 'conservative':
            adaptive_adjustments.append("Used conservative retrieval due to low domain relevance")
        if analysis.uncertainty_required:
            adaptive_adjustments.append("Uncertainty expression recommended")
        if avg_relevance < analysis.confidence_threshold:
            adaptive_adjustments.append("Low confidence context - increased uncertainty warranted")
        
        return {
            'query': query,
            'analysis': analysis,
            'retrieved_items': retrieved,
            'strategy_used': strategy_used,
            'formatted_context': formatted_context,
            'context_quality': avg_relevance,
            'context_diversity': context_diversity,
            'adaptive_adjustments': adaptive_adjustments,
            'confidence_threshold': analysis.confidence_threshold
        }
    
    def evaluate_adaptation(self, queries: List[str]) -> pd.DataFrame:
        """Evaluate adaptive pipeline on multiple queries"""
        results = []
        
        for query in queries:
            try:
                result = self.process_query(query)
                
                results.append({
                    'query': query,
                    'query_type': result['analysis'].query_type,
                    'domain_relevance': result['analysis'].domain_relevance,
                    'complexity': result['analysis'].complexity,
                    'strategy_used': result['strategy_used'],
                    'num_retrieved': len(result['retrieved_items']),
                    'context_quality': result['context_quality'],
                    'context_diversity': result['context_diversity'],
                    'num_adjustments': len(result['adaptive_adjustments']),
                    'uncertainty_required': result['analysis'].uncertainty_required
                })
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


def main():
    """Test the adaptive RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive RAG Pipeline')
    parser.add_argument('--qa-data', default='rag_input/selected_qa_pairs.json', 
                       help='Q&A pairs JSON file')
    parser.add_argument('--config', default='audio_equipment_domain_questions.json',
                       help='Domain configuration file')
    parser.add_argument('--test-queries', nargs='+', 
                       default=['What is impedance matching?', 
                               'How do you change a tire?',
                               'Compare tube vs solid-state amplifiers'],
                       help='Test queries')
    parser.add_argument('--output-dir', default='adaptive_rag_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load Q&A data
    logger.info(f"Loading Q&A data from {args.qa_data}")
    with open(args.qa_data, 'r') as f:
        qa_data = json.load(f)
    
    # Initialize pipeline
    pipeline = AdaptiveRAGPipeline(qa_data, args.config)
    
    # Process test queries
    logger.info("Processing test queries...")
    for query in args.test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info('='*60)
        
        result = pipeline.process_query(query)
        
        print(f"\nQuery Type: {result['analysis'].query_type}")
        print(f"Domain Relevance: {result['analysis'].domain_relevance:.3f}")
        print(f"Complexity: {result['analysis'].complexity}")
        print(f"Strategy Used: {result['strategy_used']}")
        print(f"Items Retrieved: {len(result['retrieved_items'])}")
        print(f"Context Quality: {result['context_quality']:.3f}")
        print(f"Adaptive Adjustments: {result['adaptive_adjustments']}")
        
        print(f"\nFormatted Context Preview:")
        print(result['formatted_context'][:500] + "..." if len(result['formatted_context']) > 500 else result['formatted_context'])
    
    # Evaluation
    logger.info("\nRunning evaluation...")
    eval_df = pipeline.evaluate_adaptation(args.test_queries)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    eval_file = output_dir / "adaptive_rag_evaluation.csv"
    eval_df.to_csv(eval_file, index=False)
    logger.info(f"Evaluation results saved to {eval_file}")
    
    # Summary
    print("\n" + "="*60)
    print("ADAPTIVE RAG PIPELINE SUMMARY")
    print("="*60)
    print(f"Queries processed: {len(args.test_queries)}")
    print(f"Average context quality: {eval_df['context_quality'].mean():.3f}")
    print(f"Strategy distribution:")
    for strategy, count in eval_df['strategy_used'].value_counts().items():
        print(f"  {strategy}: {count}")
    print(f"Query type distribution:")
    for qtype, count in eval_df['query_type'].value_counts().items():
        print(f"  {qtype}: {count}")


if __name__ == "__main__":
    main()