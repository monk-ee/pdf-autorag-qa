"""
Transcript Q&A Extractor

Specialized extractor for conversational transcripts (VTT files).
"""
import logging
import time
from typing import List, Dict, Any
import torch
from .base_extractor import BaseQAExtractor
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)

class TranscriptQAExtractor(BaseQAExtractor):
    """Q&A extractor specialized for conversational transcripts."""
    
    def load_transcript(self, filepath: str) -> str:
        """Load transcript from VTT or text file with advanced VTT parsing."""
        if filepath.endswith(".vtt"):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use TextProcessor for VTT cleaning
            content = TextProcessor.clean_vtt_text(content)
            text_segments = TextProcessor.extract_vtt_segments(content)
            reconstructed_text = TextProcessor.reconstruct_sentences(text_segments)
            
            logger.info(f"VTT processing: {len(text_segments)} segments -> reconstructed text")
            return reconstructed_text
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    
    def create_prompts(self, chunks: List[str]) -> List[str]:
        """Create prompts for Q&A extraction from conversational content."""
        prompts = []
        for chunk in chunks:
            if "Instruct" in self.model_name or "instruct" in self.model_name:
                prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Extract Q&A pairs from this transcript. Format as Q: [question] A: [answer]

{chunk}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                prompt = f"""Extract Q&A pairs. Format: Q: [question] A: [answer]

{chunk}

Q&A:"""
            prompts.append(prompt)
        return prompts
    
    def extract_qa_batch(self, chunks: List[str]) -> List[str]:
        """Extract Q&A pairs from multiple chunks using batch processing."""
        prompts = self.create_prompts(chunks)
        
        return self.process_batch(
            prompts,
            max_new_tokens=256,
            max_length=8192,
            do_sample=False
        )
    
    def process_file(self, input_file: str, output_file: str, 
                    chunk_size: int = 800, chunk_overlap: int = 200, 
                    batch_size: int = 4):
        """Process a transcript file and extract Q&A pairs."""
        start_time = time.time()
        logger.info(f"Processing transcript: {input_file}")
        
        # Load and chunk the transcript
        text = self.load_transcript(input_file)
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        
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
        for i in tqdm(range(0, len(chunks), batch_size), desc="Extracting Q&A pairs"):
            batch_start = time.time()
            batch_chunks = chunks[i:i + batch_size]
            
            # Store prompts for analysis
            batch_prompts = self.create_prompts(batch_chunks)
            all_prompts.extend(batch_prompts)
            
            # Extract Q&A pairs
            inference_start = time.time()
            qa_texts = self.extract_qa_batch(batch_chunks)
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
            'processing_stats': {
                'total_runtime': total_time,
                'model_inference_time': total_inference_time,
                'total_chunks': len(chunks),
                'batch_size': batch_size,
                'total_qa_pairs': len(all_pairs),
                'pairs_per_chunk': len(all_pairs) / len(chunks) if chunks else 0,
                'pairs_per_minute': (len(all_pairs) / total_time) * 60 if total_time > 0 else 0
            },
            'resource_utilization': resource_metrics,
            'token_performance': token_metrics,
            'extraction_type': 'transcript'
        }
        
        # Save results
        report_file = self.save_results(all_pairs, performance_metrics, output_file)
        
        # Log summary
        logger.info("="*80)
        logger.info("TRANSCRIPT Q&A EXTRACTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total Runtime: {total_time:.2f}s ({total_time/60:.1f}m)")
        logger.info(f"Chunks Processed: {len(chunks)}")
        logger.info(f"Q&A Pairs Extracted: {len(all_pairs)}")
        logger.info(f"Extraction Rate: {len(all_pairs)/total_time:.1f} pairs/sec")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Performance report: {report_file}")
        logger.info("="*80)
        
        return len(all_pairs), performance_metrics