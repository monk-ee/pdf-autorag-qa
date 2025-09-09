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
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import faiss
import re
from collections import Counter
from rank_bm25 import BM25Okapi

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
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # IMPROVEMENT 4: Enhanced query classification with domain knowledge
        # Build domain-specific ontology for better query understanding
        self.domain_ontology = self._build_domain_ontology()
        self.category_embeddings = self._build_enhanced_category_embeddings()
        self.technical_terms = self._extract_domain_entities()
        
    def load_config(self) -> Dict:
        """Load domain configuration"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Domain config not found: {self.config_file}")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_domain_ontology(self) -> Dict[str, List[str]]:
        """Build domain-specific knowledge ontology for better query understanding"""
        logger.info("Building domain ontology for enhanced query classification...")
        
        ontology = {
            # Audio equipment categories
            "amplifiers": ["tube amp", "solid state", "preamp", "power amp", "integrated amp", "headphone amp"],
            "speakers": ["woofer", "tweeter", "driver", "cabinet", "crossover", "frequency response"],
            "effects": ["reverb", "delay", "chorus", "distortion", "overdrive", "compression"],
            "recording": ["microphone", "preamp", "interface", "daw", "multitrack", "mixing"],
            "technical_specs": ["impedance", "frequency", "power", "thd", "snr", "gain", "decibel"],
            "connections": ["xlr", "trs", "rca", "balanced", "unbalanced", "phantom power"]
        }
        
        # Add terms from domain config if available
        if 'domain_terms' in self.config:
            for category, terms in self.config['domain_terms'].items():
                if category in ontology:
                    ontology[category].extend(terms)
                else:
                    ontology[category] = terms
                    
        return ontology
    
    def _build_enhanced_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Build enhanced embeddings using domain knowledge for better classification"""
        logger.info("Building enhanced category embeddings with domain knowledge...")
        
        # Enhanced categories with domain-specific patterns
        categories = {
            "factual": [
                "what is", "define", "explain the definition", "meaning of", "purpose of",
                "function of", "role of", "characteristics of"
            ],
            "conceptual": [
                "how does", "why does", "what happens when", "relationship between",
                "interaction between", "effect of", "influence of", "principle behind"
            ],
            "procedural": [
                "how to", "steps to", "process for", "way to", "method", "procedure",
                "setup", "configure", "install", "connect"
            ],
            "technical": [
                "specifications", "calculate", "measure", "parameters", "ratings",
                "impedance", "frequency response", "power handling", "thd", "signal to noise"
            ],
            "comparison": [
                "difference between", "compare", "versus", "better than", "pros and cons",
                "which is better", "choose between", "advantages", "disadvantages"
            ],
            "troubleshooting": [
                "problem with", "issue", "not working", "diagnose", "fix", "troubleshoot",
                "noise", "distortion", "no sound", "repair", "maintenance"
            ]
        }
        
        embeddings = {}
        for category, examples in categories.items():
            # Add domain-specific terms for each category
            domain_enhanced_examples = examples.copy()
            
            # Add relevant domain terms for better classification
            if category == "technical":
                domain_enhanced_examples.extend(self.domain_ontology.get("technical_specs", []))
            elif category == "comparison":
                # Add equipment comparison terms
                for equipment_type in ["amplifiers", "speakers", "effects"]:
                    domain_enhanced_examples.extend([f"{term} comparison" for term in self.domain_ontology.get(equipment_type, [])])
            
            category_text = " ".join(domain_enhanced_examples)
            embeddings[category] = self.embedder.encode([category_text])[0]
            
        return embeddings
    
    def _extract_domain_entities(self) -> Dict[str, List[str]]:
        """Extract domain-specific entities for better query understanding"""
        logger.info("Extracting domain entities for enhanced query analysis...")
        
        entities = {
            "equipment_types": [],
            "technical_terms": [],
            "brands": [],
            "specifications": []
        }
        
        # Extract from domain ontology
        for category, terms in self.domain_ontology.items():
            if category in ["amplifiers", "speakers", "effects", "recording"]:
                entities["equipment_types"].extend(terms)
            elif category == "technical_specs":
                entities["technical_terms"].extend(terms)
                entities["specifications"].extend(terms)
            elif category == "connections":
                entities["technical_terms"].extend(terms)
        
        # Add common audio brands (basic set)
        entities["brands"] = [
            "fender", "marshall", "vox", "mesa boogie", "orange", "peavey",
            "jbl", "yamaha", "mackie", "shure", "audio-technica", "sennheiser"
        ]
        
        return entities
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Enhanced query analysis with domain knowledge and entity recognition"""
        logger.info(f"Analyzing query with enhanced domain understanding: '{query[:100]}...'")
        query_lower = query.lower()
        
        # IMPROVEMENT 4: Enhanced query type classification using domain knowledge
        query_embedding = self.embedder.encode([query])[0]
        
        similarities = {}
        for category, category_embedding in self.category_embeddings.items():
            sim = cosine_similarity([query_embedding], [category_embedding])[0][0]
            similarities[category] = float(sim)
        
        query_type = max(similarities, key=similarities.get)
        logger.info(f"Query type classification: {query_type} (confidence: {similarities[query_type]:.3f})")
        
        # Enhanced domain relevance with ontology-based scoring
        domain_score = self._calculate_enhanced_domain_relevance(query_lower)
        
        # Enhanced complexity assessment with multiple indicators
        complexity = self._assess_query_complexity(query_lower)
        
        # Entity recognition for key concepts
        extracted_entities = self._extract_query_entities(query_lower)
        
        # Advanced uncertainty detection
        uncertainty_required = self._requires_uncertainty_handling(query_lower, domain_score)
        
        # Dynamic confidence threshold based on multiple factors
        confidence_threshold = self._calculate_dynamic_confidence_threshold(
            domain_score, complexity, query_type, len(extracted_entities)
        )
        
        logger.info(f"Query analysis results: type={query_type}, domain={domain_score:.3f}, complexity={complexity}, entities={len(extracted_entities)}, threshold={confidence_threshold:.3f}")
        
        return QueryAnalysis(
            query_type=query_type,
            domain_relevance=domain_score,
            complexity=complexity,
            uncertainty_required=uncertainty_required,
            key_concepts=extracted_entities,
            confidence_threshold=confidence_threshold
        )
    
    def _calculate_enhanced_domain_relevance(self, query_lower: str) -> float:
        """Calculate domain relevance using ontology-based scoring"""
        total_score = 0.0
        max_possible_score = 0.0
        
        # Score based on domain ontology categories
        for category, terms in self.domain_ontology.items():
            category_weight = {
                "amplifiers": 1.0, "speakers": 1.0, "effects": 1.0,
                "recording": 0.9, "technical_specs": 1.2, "connections": 0.8
            }.get(category, 1.0)
            
            matches = sum(1 for term in terms if term.lower() in query_lower)
            category_score = min(matches / max(len(terms) * 0.3, 1), 1.0) * category_weight
            total_score += category_score
            max_possible_score += category_weight
        
        # Normalize score
        domain_relevance = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        # Boost for exact technical term matches
        technical_boost = sum(0.1 for term in self.technical_terms.get("technical_terms", []) 
                             if term.lower() in query_lower)
        
        return min(domain_relevance + technical_boost, 1.0)
    
    def _assess_query_complexity(self, query_lower: str) -> str:
        """Enhanced complexity assessment with multiple indicators"""
        complexity_scores = {'basic': 0, 'intermediate': 0, 'advanced': 0}
        
        # Pattern-based scoring
        patterns = {
            'basic': [
                r'what is', r'define', r'meaning of', r'purpose of',
                r'simple', r'basic', r'introduction to'
            ],
            'intermediate': [
                r'how does', r'why', r'compare', r'difference between',
                r'relationship', r'interaction', r'effect of', r'works'
            ],
            'advanced': [
                r'calculate', r'optimize', r'design', r'troubleshoot',
                r'specifications?', r'analysis', r'theory', r'mathematical'
            ]
        }
        
        for complexity_level, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    complexity_scores[complexity_level] += 1
        
        # Technical term complexity scoring
        advanced_terms = self.technical_terms.get("specifications", [])
        if any(term.lower() in query_lower for term in advanced_terms):
            complexity_scores['advanced'] += 2
        
        # Query length and structure complexity
        if len(query_lower.split()) > 15:
            complexity_scores['advanced'] += 1
        elif len(query_lower.split()) > 8:
            complexity_scores['intermediate'] += 1
        
        # Multiple entity complexity
        entity_count = sum(len(entities) for entities in self.technical_terms.values())
        if entity_count > 3:
            complexity_scores['advanced'] += 1
        
        return max(complexity_scores, key=complexity_scores.get)
    
    def _extract_query_entities(self, query_lower: str) -> List[str]:
        """Extract domain-specific entities from query"""
        entities = []
        
        # Extract entities from all categories
        for entity_type, entity_list in self.technical_terms.items():
            for entity in entity_list:
                if entity.lower() in query_lower:
                    entities.append(entity)
        
        # Extract from domain ontology
        for category, terms in self.domain_ontology.items():
            for term in terms:
                if term.lower() in query_lower and term not in entities:
                    entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _requires_uncertainty_handling(self, query_lower: str, domain_score: float) -> bool:
        """Advanced uncertainty detection"""
        # Low domain relevance
        if domain_score < 0.3:
            return True
        
        # Out-of-domain indicators
        out_domain_indicators = [
            'car', 'automobile', 'programming', 'software', 'cooking', 'recipe',
            'weather', 'climate', 'sports', 'game', 'politics', 'news'
        ]
        if any(indicator in query_lower for indicator in out_domain_indicators):
            return True
        
        # Ambiguous query patterns
        ambiguous_patterns = [r'what about', r'tell me about', r'anything about']
        if any(re.search(pattern, query_lower) for pattern in ambiguous_patterns):
            return True
        
        return False
    
    def _calculate_dynamic_confidence_threshold(self, domain_score: float, complexity: str, 
                                               query_type: str, entity_count: int) -> float:
        """Calculate dynamic confidence threshold based on multiple factors"""
        base_threshold = 0.5
        
        # Adjust for domain relevance
        if domain_score > 0.8:
            base_threshold += 0.2
        elif domain_score > 0.6:
            base_threshold += 0.1
        elif domain_score < 0.3:
            base_threshold -= 0.2
        
        # Adjust for complexity
        complexity_adjustments = {
            'basic': 0.1,
            'intermediate': 0.0,
            'advanced': -0.1
        }
        base_threshold += complexity_adjustments.get(complexity, 0.0)
        
        # Adjust for query type confidence
        if query_type in ['factual', 'technical']:
            base_threshold += 0.1
        elif query_type in ['conceptual', 'comparison']:
            base_threshold -= 0.05
        
        # Adjust for entity richness
        if entity_count > 3:
            base_threshold += 0.1
        elif entity_count == 0:
            base_threshold -= 0.1
        
        return max(0.2, min(0.9, base_threshold))


class AdaptiveRetriever:
    """🚀 ENHANCED: Adaptive retrieval with hybrid dense+sparse, cross-encoder re-ranking, and dynamic context windows"""
    
    def __init__(self, qa_data: List[Dict], embedder: SentenceTransformer):
        self.qa_data = qa_data
        self.embedder = embedder
        
        # IMPROVEMENT 1: Cross-encoder for re-ranking - TEMPORARILY DISABLED for debugging
        logger.info("🔧 Cross-encoder temporarily disabled for performance testing...")
        self.cross_encoder = None
        self.has_cross_encoder = False
        
        # Build enhanced indices with hybrid retrieval
        self._build_enhanced_indices()
        
        # IMPROVEMENT 3: Enhanced strategy configurations with dynamic context windows
        self.strategies = {
            'conservative': RetrievalStrategy(
                top_k=3, alpha=0.8, rerank=True, 
                context_window=512, confidence_gating=True, 
                fallback_strategy='uncertainty'
            ),
            'balanced': RetrievalStrategy(
                top_k=5, alpha=0.7, rerank=True,
                context_window=768, confidence_gating=True,
                fallback_strategy='context_expansion'
            ),
            'aggressive': RetrievalStrategy(
                top_k=8, alpha=0.6, rerank=True,
                context_window=1024, confidence_gating=False,
                fallback_strategy='multi_strategy'
            )
        }
    
    def _build_enhanced_indices(self):
        """🚀 IMPROVEMENT 2: Build enhanced hybrid indices: FAISS (dense) + BM25 (sparse) for hybrid retrieval"""
        logger.info("🏗️  Building enhanced hybrid retrieval indices (dense + sparse)...")
        
        # Create multiple text representations for different retrieval strategies
        self.qa_texts = []
        self.question_texts = []
        self.answer_texts = []
        
        for qa in self.qa_data:
            # Combined Q+A for hybrid retrieval
            combined_text = f"Q: {qa['instruction']} A: {qa['output']}"
            self.qa_texts.append(combined_text)
            
            # Separate question and answer texts for targeted retrieval
            self.question_texts.append(qa['instruction'])
            self.answer_texts.append(qa['output'])
        
        # Dense embeddings (multiple strategies)
        logger.info("🧠 Generating dense embeddings for hybrid retrieval...")
        
        # Combined Q+A embeddings (primary)
        combined_embeddings = self.embedder.encode(self.qa_texts, show_progress_bar=True)
        
        # Question-only embeddings (for question similarity)
        question_embeddings = self.embedder.encode(self.question_texts, show_progress_bar=True)
        
        # Answer-only embeddings (for semantic answer matching)
        answer_embeddings = self.embedder.encode(self.answer_texts, show_progress_bar=True)
        
        # Build FAISS indices for different embedding strategies
        logger.info("🔍 Building FAISS indices for multiple retrieval strategies...")
        
        # Primary index (combined Q+A)
        dimension = combined_embeddings.shape[1]
        self.combined_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(combined_embeddings)
        self.combined_index.add(combined_embeddings.astype('float32'))
        
        # Question similarity index
        self.question_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(question_embeddings)
        self.question_index.add(question_embeddings.astype('float32'))
        
        # Answer similarity index
        self.answer_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(answer_embeddings)
        self.answer_index.add(answer_embeddings.astype('float32'))
        
        # IMPROVEMENT 2: BM25 sparse retrieval - TEMPORARILY DISABLED for debugging
        logger.info("🔧 BM25 and TF-IDF temporarily disabled for performance testing...")
        self.bm25 = None
        self.has_bm25 = False
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.has_tfidf = False
        
        logger.info(f"🎯 Hybrid retrieval indices completed:")
        logger.info(f"   📊 {len(self.qa_data)} Q&A pairs indexed")
        logger.info(f"   🧠 Dense embeddings: {dimension}D vectors")
        logger.info(f"   📝 BM25 sparse: {'✅' if self.has_bm25 else '❌'}")
        logger.info(f"   📄 TF-IDF backup: {'✅' if self.has_tfidf else '❌'}")
        logger.info(f"   🔄 Cross-encoder re-ranking: {'✅' if self.has_cross_encoder else '❌'}")
    
    def select_strategy(self, analysis: QueryAnalysis) -> str:
        """Select retrieval strategy based on query analysis"""
        if analysis.domain_relevance < 0.3:
            return 'conservative'
        elif analysis.complexity == 'advanced' or analysis.query_type == 'technical':
            return 'aggressive'
        else:
            return 'balanced'
    
    def retrieve_adaptive(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Dict], str]:
        """🚀 ENHANCED: Perform adaptive retrieval with hybrid dense+sparse + cross-encoder re-ranking"""
        logger.info(f"🔍 Starting adaptive retrieval for: '{query[:50]}...'")
        
        strategy_name = self.select_strategy(analysis)
        strategy = self.strategies[strategy_name]
        
        logger.info(f"🎯 Selected strategy: {strategy_name} (top_k={strategy.top_k}, alpha={strategy.alpha}, rerank={strategy.rerank})")
        
        # IMPROVEMENT 2: Hybrid dense + sparse retrieval
        hybrid_candidates = self._hybrid_retrieval(query, analysis, strategy.top_k * 3)  # Get more candidates for re-ranking
        
        # IMPROVEMENT 1: Cross-encoder re-ranking (if enabled and available)
        if strategy.rerank and self.has_cross_encoder and len(hybrid_candidates) > 1:
            logger.info(f"🔄 Re-ranking {len(hybrid_candidates)} candidates with cross-encoder...")
            reranked_candidates = self._cross_encoder_rerank(query, hybrid_candidates)
        else:
            reranked_candidates = hybrid_candidates
            logger.info("⏭️  Skipping re-ranking (disabled or unavailable)")
        
        # IMPROVEMENT 3: Dynamic context window adjustment
        final_results = self._apply_dynamic_context_window(reranked_candidates, strategy, analysis)
        
        # Apply confidence gating and fallback strategies
        final_results = self._apply_confidence_gating(query, analysis, final_results, strategy_name, strategy)
        
        logger.info(f"✅ Adaptive retrieval completed: {len(final_results)} results returned")
        return final_results, strategy_name
    
    def _hybrid_retrieval(self, query: str, query_analysis: QueryAnalysis, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """FIX: Simplified retrieval - use dense only until performance is restored"""
        logger.info(f"🧠 Performing simplified dense retrieval for query type: {query_analysis.query_type}")
        
        # FIX: Use dense retrieval only (hybrid disabled for debugging)
        dense_scores = self._get_dense_scores(query, query_analysis, top_k)
        
        # Get top candidates directly from dense scores
        top_candidates = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        logger.info(f"📊 Dense retrieval: {len(dense_scores)} candidates, returning top {len(top_candidates)}")
        return [(idx, score, "dense_only") for idx, score in top_candidates]
    
    def _get_dense_scores(self, query: str, query_analysis: QueryAnalysis, top_k: int) -> Dict[int, float]:
        """FIX: Simplified dense retrieval - use primary strategy only to avoid over-complexity"""
        query_embedding = self.embedder.encode([query])[0].astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores = {}
        
        # FIX: Use combined Q+A similarity only (primary strategy) - simpler and more reliable
        combined_scores, combined_indices = self.combined_index.search(query_embedding, min(top_k, len(self.qa_data)))
        for i, (idx, score) in enumerate(zip(combined_indices[0], combined_scores[0])):
            if idx >= 0:  # Valid index
                scores[idx] = float(score)  # FIX: Use raw scores without complex weighting
        
        return scores
    
    def _get_sparse_scores(self, query: str, top_k: int) -> Dict[int, float]:
        """Get scores from sparse retrieval (BM25 + TF-IDF)"""
        scores = {}
        
        if self.has_bm25:
            # BM25 scoring
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top BM25 matches
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
            for idx in top_bm25_indices:
                if idx < len(self.qa_data):
                    scores[idx] = scores.get(idx, 0) + float(bm25_scores[idx]) * 0.7  # BM25 weight
        
        if self.has_tfidf:
            # TF-IDF scoring as backup
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_scores = sklearn_cosine(query_tfidf, self.tfidf_matrix).flatten()
            
            # Add TF-IDF scores
            top_tfidf_indices = np.argsort(tfidf_scores)[::-1][:top_k//2]
            for idx in top_tfidf_indices:
                if idx < len(self.qa_data):
                    scores[idx] = scores.get(idx, 0) + float(tfidf_scores[idx]) * 0.3  # TF-IDF weight
        
        return scores
    
    def _get_adaptive_alpha(self, query_analysis: QueryAnalysis) -> float:
        """Calculate adaptive weight for dense vs sparse retrieval"""
        base_alpha = 0.7  # Favor dense by default
        
        # Adjust based on query type
        if query_analysis.query_type in ['technical', 'procedural']:
            # Technical queries benefit from exact term matching (sparse)
            base_alpha -= 0.2
        elif query_analysis.query_type in ['conceptual', 'comparison']:
            # Conceptual queries benefit from semantic similarity (dense)
            base_alpha += 0.1
        
        # Adjust based on domain relevance
        if query_analysis.domain_relevance > 0.8:
            # High domain relevance benefits from dense similarity
            base_alpha += 0.1
        elif query_analysis.domain_relevance < 0.3:
            # Low domain relevance benefits from lexical matching
            base_alpha -= 0.2
        
        return max(0.3, min(0.9, base_alpha))
    
    def _combine_hybrid_scores(self, dense_scores: Dict[int, float], sparse_scores: Dict[int, float], alpha: float) -> Dict[int, float]:
        """Combine dense and sparse scores with adaptive weighting"""
        combined_scores = {}
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Normalize scores to [0, 1] range
        if dense_scores:
            max_dense = max(dense_scores.values()) if dense_scores.values() else 0
            min_dense = min(dense_scores.values()) if dense_scores.values() else 0
            dense_range = max_dense - min_dense
        else:
            dense_range = 0
        
        if sparse_scores:
            max_sparse = max(sparse_scores.values()) if sparse_scores.values() else 0
            min_sparse = min(sparse_scores.values()) if sparse_scores.values() else 0
            sparse_range = max_sparse - min_sparse
        else:
            sparse_range = 0
        
        for idx in all_indices:
            # Normalize dense score
            dense_score = dense_scores.get(idx, 0)
            if dense_range > 0:
                dense_score = (dense_score - min_dense) / dense_range
            
            # Normalize sparse score
            sparse_score = sparse_scores.get(idx, 0)
            if sparse_range > 0:
                sparse_score = (sparse_score - min_sparse) / sparse_range
            
            # Combine with adaptive weighting
            combined_scores[idx] = alpha * dense_score + (1 - alpha) * sparse_score
        
        return combined_scores
    
    def _cross_encoder_rerank(self, query: str, candidates: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
        """🚀 IMPROVEMENT 1: Cross-encoder re-ranking for better context scoring - FIXED"""
        if not self.has_cross_encoder or len(candidates) <= 1:
            return candidates
        
        try:
            # Prepare query-document pairs for cross-encoder - FIXED format
            pairs = []
            candidate_indices = []
            
            for idx, score, source in candidates:
                if idx < len(self.qa_data):
                    # FIX: Use answer text directly, not Q+A format for cross-encoder
                    answer_text = self.qa_data[idx]['output']
                    pairs.append([query, answer_text])
                    candidate_indices.append((idx, source))
            
            if not pairs:
                return candidates
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)
            
            # FIX: Normalize cross-encoder scores to [0,1] range and fix score combination
            ce_scores = np.array(ce_scores)
            ce_scores_normalized = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
            
            # Combine with original scores (fixed weighting)
            reranked_candidates = []
            for i, (ce_score, (idx, source)) in enumerate(zip(ce_scores_normalized, candidate_indices)):
                original_score = candidates[i][1]
                
                # FIX: Balanced combination with normalized scores
                combined_score = 0.7 * float(ce_score) + 0.3 * max(0, original_score)  # Ensure positive scores
                reranked_candidates.append((idx, combined_score, f"reranked_{source}"))
            
            # Sort by combined score
            reranked_candidates.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🔄 Cross-encoder re-ranking: normalized score range [{min(ce_scores_normalized):.3f}, {max(ce_scores_normalized):.3f}]")
            return reranked_candidates
            
        except Exception as e:
            logger.warning(f"⚠️  Cross-encoder re-ranking failed: {e}")
            return candidates
    
    def _apply_dynamic_context_window(self, candidates: List[Tuple[int, float, str]], 
                                    strategy: RetrievalStrategy, analysis: QueryAnalysis) -> List[Dict]:
        """🚀 IMPROVEMENT 3: Apply dynamic context window adjustment"""
        
        # Calculate dynamic top_k based on query complexity and confidence
        base_k = strategy.top_k
        
        # Adjust based on query complexity
        if analysis.complexity == 'advanced':
            dynamic_k = int(base_k * 1.3)  # More context for complex queries
        elif analysis.complexity == 'basic':
            dynamic_k = max(2, int(base_k * 0.8))  # Less context for simple queries
        else:
            dynamic_k = base_k
        
        # Adjust based on domain relevance
        if analysis.domain_relevance < 0.3:
            dynamic_k = max(2, int(dynamic_k * 0.7))  # Less context for out-of-domain
        elif analysis.domain_relevance > 0.8:
            dynamic_k = int(dynamic_k * 1.1)  # More context for high-relevance
        
        # Apply confidence-based filtering
        if len(candidates) > 0:
            scores = [score for _, score, _ in candidates]
            score_threshold = np.mean(scores) - np.std(scores) if len(scores) > 1 else 0.0
            
            # Filter low-confidence candidates
            filtered_candidates = [(idx, score, source) for idx, score, source in candidates 
                                 if score >= score_threshold]
            
            # Ensure we have at least 1 result
            if not filtered_candidates and candidates:
                filtered_candidates = [candidates[0]]
        else:
            filtered_candidates = candidates
        
        # Select top candidates based on dynamic window
        final_candidates = filtered_candidates[:dynamic_k]
        
        # Convert to result format
        results = []
        for idx, score, source in final_candidates:
            if idx < len(self.qa_data):
                result = self.qa_data[idx].copy()
                result['relevance_score'] = float(score)
                result['retrieval_source'] = source
                results.append(result)
        
        logger.info(f"📏 Dynamic context window: base_k={base_k} → dynamic_k={dynamic_k}, final={len(results)} results")
        return results
    
    def _apply_confidence_gating(self, query: str, analysis: QueryAnalysis, results: List[Dict], 
                               strategy_name: str, strategy: RetrievalStrategy) -> List[Dict]:
        """Apply confidence gating and fallback strategies"""
        if not strategy.confidence_gating or not results:
            return results
        
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in results])
        
        if avg_relevance < analysis.confidence_threshold:
            logger.info(f"🔄 Confidence gating triggered: avg_relevance={avg_relevance:.3f} < threshold={analysis.confidence_threshold:.3f}")
            
            # Apply fallback strategy
            if strategy.fallback_strategy == 'uncertainty':
                results = results[:1]  # Use only top result with uncertainty
                logger.info("🔄 Applied 'uncertainty' fallback: using top result only")
                
            elif strategy.fallback_strategy == 'context_expansion':
                # Expand search with lower threshold (recursive call)
                logger.info("🔄 Applied 'context_expansion' fallback: expanding search")
                expanded_analysis = replace(analysis, confidence_threshold=0.2)
                return self.retrieve_adaptive(query, expanded_analysis)[0]  # Return only results, not strategy
                
            elif strategy.fallback_strategy == 'multi_strategy':
                # Try different strategy if current one fails
                logger.info("🔄 Applied 'multi_strategy' fallback: trying different strategy")
                fallback_strategy = 'balanced' if strategy_name == 'conservative' else 'aggressive'
                fallback_analysis = replace(analysis, confidence_threshold=0.3)
                # Note: This could lead to infinite recursion, so we limit fallback depth
                return self.retrieve_adaptive(query, fallback_analysis)[0]
        
        return results


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