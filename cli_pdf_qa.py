#!/usr/bin/env python3
"""
CLI for PDF Q&A Generation using the common library.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the library to the path
sys.path.insert(0, str(Path(__file__).parent))

from qa_extraction_lib import PDFQAGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from PDF content using the common library")
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("-o", "--output", default="pdf_qa_output.jsonl", help="Output JSONL file")
    parser.add_argument("-m", "--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face model name")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in words")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum total sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling")
    parser.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling")
    
    # Debug parameters
    parser.add_argument("--debug-prompts", action="store_true", help="Save generated prompts to file")
    parser.add_argument("--debug-chunks", action="store_true", help="Save chunk information to file")
    parser.add_argument("--run-name", default="", help="Run name for output files")
    
    # Prompt configuration
    parser.add_argument("--prompts-file", default="prompts.json", help="JSON file containing prompt templates")
    parser.add_argument("--custom-template", help="Custom prompt template (overrides templates from file)")
    parser.add_argument("--difficulty-levels", nargs="+", default=["basic", "intermediate", "advanced"], 
                       help="Difficulty levels to cycle through")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize generator with the common library
    try:
        generator = PDFQAGenerator(
            model_name=args.model,
            use_gpu=not args.no_gpu,
            quantize=args.quantize,
            prompts_file=args.prompts_file
        )
        
        # Set generation parameters
        generation_params = {
            'max_new_tokens': args.max_new_tokens,
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'do_sample': args.do_sample
        }
        
        logger.info(f"Generation parameters: {generation_params}")
        
        # Process PDF with custom parameters
        num_pairs, performance_metrics = generator.process_file_with_params(
            args.input_file,
            args.output,
            args.chunk_size,
            args.chunk_overlap,
            args.batch_size,
            generation_params,
            debug_prompts=args.debug_prompts,
            debug_chunks=args.debug_chunks,
            run_name=args.run_name,
            custom_template=args.custom_template,
            difficulty_levels=args.difficulty_levels
        )
        
        logger.info(f"Processing complete. Generated {num_pairs} Q&A pairs.")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())