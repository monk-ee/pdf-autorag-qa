"""
PDF Q&A Generator

Specialized generator for creating educational Q&A pairs from PDF content.
"""
import os
import logging
import time
from typing import List, Dict, Any
import torch
import PyPDF2
import fitz  # PyMuPDF
from .base_extractor import BaseQAExtractor
from .text_processing import TextProcessor
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class PDFQAGenerator(BaseQAExtractor):
    """Q&A generator specialized for PDF educational content."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 use_gpu: bool = True, quantize: bool = False, prompts_file: str = "prompts.json"):
        super().__init__(model_name, use_gpu, quantize)
        self.prompt_manager = PromptManager(prompts_file)
        logger.info(f"Initialized PDF Q&A Generator with prompt file: {prompts_file}")
    
    def extract_pdf_content(self, filepath: str) -> Dict[str, Any]:
        """Extract structured content from PDF with metadata."""
        logger.info(f"Extracting content from PDF: {filepath}")
        
        content_data = {
            'raw_text': '',
            'pages': [],
            'metadata': {},
            'structure': {
                'sections': [],
                'figures': [],
                'tables': []
            }
        }
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(filepath)
            
            # Extract metadata
            content_data['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'page_count': len(doc),
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
            }
            
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract images and figures metadata
                image_list = page.get_images()
                figures = []
                for img_index, img in enumerate(image_list):
                    figures.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'xref': img[0]
                    })
                
                # Extract text blocks with positioning
                blocks = page.get_text("dict")
                sections = TextProcessor.analyze_text_structure(blocks, page_num + 1)
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'figures': figures,
                    'sections': sections
                }
                
                content_data['pages'].append(page_data)
                full_text.append(text)
                content_data['structure']['figures'].extend(figures)
                content_data['structure']['sections'].extend(sections)
            
            doc.close()
            
            # Combine and clean text
            content_data['raw_text'] = '\n\n'.join(full_text)
            content_data['cleaned_text'] = TextProcessor.clean_pdf_text(content_data['raw_text'])
            
            logger.info(f"PDF extraction complete: {len(content_data['pages'])} pages, "
                       f"{len(content_data['cleaned_text'].split())} words")
            
            return content_data
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            # Fallback to PyPDF2
            return self._fallback_pdf_extraction(filepath)
    
    def _fallback_pdf_extraction(self, filepath: str) -> Dict[str, Any]:
        """Fallback PDF extraction using PyPDF2."""
        logger.warning("Using fallback PDF extraction method")
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                pages = []
                full_text = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    full_text.append(text)
                
                raw_text = '\n\n'.join(full_text)
                
                return {
                    'raw_text': raw_text,
                    'cleaned_text': TextProcessor.clean_pdf_text(raw_text),
                    'pages': pages,
                    'metadata': {
                        'page_count': len(pages),
                        'extraction_method': 'PyPDF2_fallback'
                    },
                    'structure': {'sections': [], 'figures': [], 'tables': []}
                }
                
        except Exception as e:
            logger.error(f"Fallback PDF extraction failed: {e}")
            raise
    
    def chunk_content_intelligently(self, content_data: Dict, chunk_size: int = 800, 
                                  overlap: int = 200) -> List[Dict]:
        """Create intelligent chunks preserving content structure."""
        text = content_data['cleaned_text']
        structure_info = content_data['structure']['sections']
        
        return TextProcessor.intelligent_chunking(text, chunk_size, overlap, structure_info)
    
    def create_prompts(self, chunks: List[Dict], difficulty_levels: List[str] = None, 
                      custom_template: str = None) -> List[str]:
        """Create prompts for generating educational Q&A pairs using prompt manager."""
        if difficulty_levels is None:
            difficulty_levels = ['basic', 'intermediate', 'advanced']
        
        return self.prompt_manager.create_prompts_for_chunks(
            task="pdf_qa_generation",
            model_name=self.model_name,
            chunks=chunks,
            difficulty_levels=difficulty_levels,
            custom_template=custom_template
        )
    
    def generate_qa_batch(self, chunks: List[Dict]) -> List[str]:
        """Generate Q&A pairs from multiple chunks using batch processing."""
        prompts = self.create_prompts(chunks)
        
        return self.process_batch(
            prompts,
            max_new_tokens=512,  # Longer for educational content
            do_sample=True,
            temperature=0.7,  # Some creativity for question generation
            top_p=0.9
        )
    
    def process_file(self, input_file: str, output_file: str, 
                    chunk_size: int = 800, chunk_overlap: int = 200, 
                    batch_size: int = 4):
        """Process a PDF file and generate Q&A pairs."""
        start_time = time.time()
        logger.info(f"Processing PDF: {input_file}")
        
        # Extract PDF content with structure
        content_data = self.extract_pdf_content(input_file)
        
        # Create intelligent chunks
        chunks = self.chunk_content_intelligently(content_data, chunk_size, chunk_overlap)
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        all_pairs = []
        batch_timings = []
        parsing_stats_list = []
        all_prompts = []
        all_responses = []
        total_inference_time = 0
        
        # Reset GPU memory tracking
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Process chunks in batches
        from tqdm import tqdm
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating Q&A pairs"):
            batch_start = time.time()
            batch_chunks = chunks[i:i + batch_size]
            
            # Store prompts for analysis
            batch_prompts = self.create_prompts(batch_chunks)
            all_prompts.extend(batch_prompts)
            
            # Generate Q&A pairs
            inference_start = time.time()
            qa_texts = self.generate_qa_batch(batch_chunks)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            all_responses.extend(qa_texts)
            
            # Parse responses
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(qa_texts))) as executor:
                parse_results = list(executor.map(self.parse_qa_output, qa_texts))
            
            # Process results
            for pairs, parsing_stats in parse_results:
                all_pairs.extend(pairs)
                parsing_stats_list.append(parsing_stats)
            
            batch_time = time.time() - batch_start
            batch_timings.append(batch_time)
            
            # Clear CUDA cache periodically
            if i % (batch_size * 5) == 0 and self.use_gpu:
                torch.cuda.empty_cache()
        
        # Calculate metrics
        total_time = time.time() - start_time
        resource_metrics = self.calculate_resource_metrics(start_time, input_file)
        token_metrics = self.calculate_token_metrics(all_prompts, all_responses, total_inference_time)
        
        # Create performance report
        performance_metrics = {
            'model_name': self.model_name,
            'input_file': input_file,
            'pdf_metadata': content_data['metadata'],
            'processing_stats': {
                'total_runtime': total_time,
                'model_inference_time': total_inference_time,
                'total_chunks': len(chunks),
                'batch_size': batch_size,
                'total_qa_pairs': len(all_pairs),
                'pairs_per_chunk': len(all_pairs) / len(chunks) if chunks else 0,
                'pairs_per_minute': (len(all_pairs) / total_time) * 60 if total_time > 0 else 0
            },
            'content_analysis': {
                'source_pages': len(content_data['pages']),
                'total_words': len(content_data['cleaned_text'].split()),
                'structure_elements': len(content_data['structure']['sections'])
            },
            'resource_utilization': resource_metrics,
            'token_performance': token_metrics,
            'generation_type': 'pdf_educational'
        }
        
        # Save results
        report_file = self.save_results(all_pairs, performance_metrics, output_file)
        
        # Log summary
        logger.info("="*80)
        logger.info("PDF Q&A GENERATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total Runtime: {total_time:.2f}s ({total_time/60:.1f}m)")
        logger.info(f"PDF Pages: {len(content_data['pages'])}")
        logger.info(f"Chunks Processed: {len(chunks)}")
        logger.info(f"Q&A Pairs Generated: {len(all_pairs)}")
        logger.info(f"Generation Rate: {len(all_pairs)/total_time:.1f} pairs/sec")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Performance report: {report_file}")
        logger.info("="*80)
        
        return len(all_pairs), performance_metrics
    
    def process_file_with_params(self, input_file: str, output_file: str, 
                                chunk_size: int = 800, chunk_overlap: int = 200, 
                                batch_size: int = 4, generation_params: dict = None,
                                debug_prompts: bool = False, debug_chunks: bool = False,
                                run_name: str = "", custom_template: str = None,
                                difficulty_levels: List[str] = None):
        """Process a PDF file with custom generation parameters and debug options."""
        import json
        from pathlib import Path
        
        start_time = time.time()
        logger.info(f"Processing PDF: {input_file}")
        logger.info(f"Generation parameters: {generation_params}")
        
        # Extract PDF content with structure
        content_data = self.extract_pdf_content(input_file)
        
        # Create intelligent chunks
        chunks = self.chunk_content_intelligently(content_data, chunk_size, chunk_overlap)
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        # Debug: Save chunk information
        if debug_chunks:
            debug_chunks_file = Path(output_file).with_suffix(f'.chunks{run_name}.json')
            debug_chunks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_chunks_file, 'w') as f:
                json.dump({
                    'total_chunks': len(chunks),
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'chunks': [
                        {
                            'index': i,
                            'word_count': chunk.get('word_count', len(chunk['text'].split())),
                            'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                        } for i, chunk in enumerate(chunks[:10])  # First 10 chunks
                    ]
                }, f, indent=2)
            logger.info(f"Debug chunks saved to: {debug_chunks_file}")
        
        all_pairs = []
        batch_timings = []
        parsing_stats_list = []
        all_prompts = []
        all_responses = []
        total_inference_time = 0
        
        # Reset GPU memory tracking
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Process chunks in batches
        from tqdm import tqdm
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating Q&A pairs"):
            batch_start = time.time()
            batch_chunks = chunks[i:i + batch_size]
            
            # Store prompts for analysis with custom parameters
            batch_prompts = self.create_prompts(batch_chunks, difficulty_levels, custom_template)
            all_prompts.extend(batch_prompts)
            
            # Debug: Log first prompt
            if i == 0:
                logger.info(f"Sample prompt (first chunk):\n{batch_prompts[0][:500]}...")
            
            # Generate Q&A pairs with custom parameters
            inference_start = time.time()
            if generation_params:
                qa_texts = self.process_batch(batch_prompts, **generation_params)
            else:
                qa_texts = self.generate_qa_batch(batch_chunks)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            all_responses.extend(qa_texts)
            
            # Debug: Log first response
            if i == 0:
                logger.info(f"Sample response (first chunk):\n{qa_texts[0][:300]}...")
            
            # Parse responses
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(qa_texts))) as executor:
                parse_results = list(executor.map(self.parse_qa_output, qa_texts))
            
            # Process results
            for pairs, parsing_stats in parse_results:
                all_pairs.extend(pairs)
                parsing_stats_list.append(parsing_stats)
            
            batch_time = time.time() - batch_start
            batch_timings.append(batch_time)
            
            logger.debug(f"Batch {i//batch_size + 1}: Generated {sum(len(pairs) for pairs, _ in parse_results)} pairs in {batch_time:.2f}s")
            
            # Clear CUDA cache periodically
            if i % (batch_size * 5) == 0 and self.use_gpu:
                torch.cuda.empty_cache()
        
        # Debug: Save prompts
        if debug_prompts:
            debug_prompts_file = Path(output_file).with_suffix(f'.prompts{run_name}.json')
            debug_prompts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_prompts_file, 'w') as f:
                json.dump({
                    'generation_params': generation_params,
                    'total_prompts': len(all_prompts),
                    'prompts': all_prompts[:5],  # First 5 prompts
                    'responses': all_responses[:5]  # First 5 responses
                }, f, indent=2)
            logger.info(f"Debug prompts saved to: {debug_prompts_file}")
        
        # Calculate metrics
        total_time = time.time() - start_time
        resource_metrics = self.calculate_resource_metrics(start_time, input_file)
        token_metrics = self.calculate_token_metrics(all_prompts, all_responses, total_inference_time)
        
        # Create performance report
        performance_metrics = {
            'model_name': self.model_name,
            'input_file': input_file,
            'generation_parameters': generation_params,
            'run_name': run_name,
            'pdf_metadata': content_data['metadata'],
            'processing_stats': {
                'total_runtime': total_time,
                'model_inference_time': total_inference_time,
                'total_chunks': len(chunks),
                'batch_size': batch_size,
                'total_qa_pairs': len(all_pairs),
                'pairs_per_chunk': len(all_pairs) / len(chunks) if chunks else 0,
                'pairs_per_minute': (len(all_pairs) / total_time) * 60 if total_time > 0 else 0
            },
            'content_analysis': {
                'source_pages': len(content_data['pages']),
                'total_words': len(content_data['cleaned_text'].split()),
                'structure_elements': len(content_data['structure']['sections'])
            },
            'resource_utilization': resource_metrics,
            'token_performance': token_metrics,
            'generation_type': 'pdf_educational'
        }
        
        # Save results
        report_file = self.save_results(all_pairs, performance_metrics, output_file)
        
        # Log summary
        logger.info("="*80)
        logger.info(f"PDF Q&A GENERATION SUMMARY - {run_name}")
        logger.info("="*80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Generation params: {generation_params}")
        logger.info(f"Total Runtime: {total_time:.2f}s ({total_time/60:.1f}m)")
        logger.info(f"PDF Pages: {len(content_data['pages'])}")
        logger.info(f"Chunks Processed: {len(chunks)}")
        logger.info(f"Q&A Pairs Generated: {len(all_pairs)}")
        logger.info(f"Generation Rate: {len(all_pairs)/total_time:.1f} pairs/sec")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Performance report: {report_file}")
        logger.info("="*80)
        
        return len(all_pairs), performance_metrics