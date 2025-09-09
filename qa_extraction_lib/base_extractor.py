"""
Base Q&A Extractor Class

Common functionality shared between transcript extraction and PDF generation.
"""
import os
import json
import logging
import time
import psutil
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from datasets import Dataset

logger = logging.getLogger(__name__)

class BaseQAExtractor(ABC):
    """Base class for Q&A extraction and generation."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 use_gpu: bool = True, quantize: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        logger.info(f"Initializing model {model_name} on device: {self.device}")
        
        # Configure torch compilation for stability
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.cache_size_limit = 1
        
        # Load model with optimizations
        self._load_model(quantize)
        self._setup_pipeline()
        self._optimize_model()
        
        logger.info("Model initialization complete")
    
    def _load_model(self, quantize: bool):
        """Load the Hugging Face model with optimizations."""
        model_kwargs = {
            "dtype": torch.float16 if self.use_gpu else torch.float32,
            "device_map": "auto" if self.use_gpu else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if "GPTQ" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        elif quantize and self.use_gpu:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        self.original_model = self.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set left padding for decoder-only models
        self.tokenizer.padding_side = 'left'
    
    def _setup_pipeline(self):
        """Setup the text generation pipeline."""
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dtype": torch.float16 if self.use_gpu else torch.float32,
            "batch_size": 16 if self.use_gpu else 1
        }
        
        if not hasattr(self.model, 'hf_device_map'):
            pipeline_kwargs["device"] = 0 if self.use_gpu else -1
            
        self.generator = pipeline("text-generation", **pipeline_kwargs)
    
    def _optimize_model(self):
        """Apply performance optimizations."""
        # Compile model for faster inference
        if hasattr(torch, 'compile') and self.use_gpu:
            self.model = torch.compile(
                self.model, 
                mode="max-autotune", 
                disable=["triton.cudagraph_trees"]
            )
            logger.info("Model compiled for faster inference (no CUDA graphs)")
        
        # Set tensor float precision for better performance
        if self.use_gpu:
            torch.set_float32_matmul_precision('high')
            logger.info("Set TensorFloat-32 precision for better performance")
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    @abstractmethod
    def create_prompts(self, chunks: List[str]) -> List[str]:
        """Create prompts for the specific task (extraction vs generation)."""
        pass
    
    def process_batch(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Process a batch of prompts using the pipeline."""
        # Create dataset for optimal parallel batch processing
        dataset = Dataset.from_dict({"text": prompts})
        
        # Process with dataset for true parallelism
        dataset_input = dataset["text"]
        if hasattr(dataset_input, 'to_list'):
            dataset_input = dataset_input.to_list()
        elif hasattr(dataset_input, '__iter__') and not isinstance(dataset_input, str):
            dataset_input = list(dataset_input)
        
        # Default generation parameters
        default_kwargs = {
            "max_new_tokens": 256,
            "max_length": 8192,
            "num_return_sequences": 1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "return_full_text": False,
            "truncation": True,
            "padding": True,
            "batch_size": len(prompts)
        }
        default_kwargs.update(generation_kwargs)
        
        outputs = self.generator(dataset_input, **default_kwargs)
        
        # Extract responses
        responses = []
        for output in outputs:
            if isinstance(output, list):
                response = output[0]['generated_text']
            else:
                response = output['generated_text']
            responses.append(response.strip())
        
        return responses
    
    def parse_qa_output(self, text: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Parse the model output into structured Q&A pairs with quality metrics."""
        lines = text.strip().splitlines()
        qa_pairs = []
        current_q = None
        current_a = None
        parsing_stats = {
            'raw_response_length': len(text),
            'total_lines': len(lines),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'q_lines': sum(1 for line in lines if line.strip().startswith("Q:")),
            'a_lines': sum(1 for line in lines if line.strip().startswith("A:")),
            'unparseable_lines': 0,
            'incomplete_pairs': 0
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                if current_q and not current_a:
                    parsing_stats['incomplete_pairs'] += 1
                current_q = line[2:].strip()
            elif line.startswith("A:"):
                current_a = line[2:].strip()
                if current_q and current_a:
                    qa_pairs.append({
                        "instruction": current_q,
                        "input": "",
                        "output": current_a,
                        "metadata": {
                            "question_length": len(current_q),
                            "answer_length": len(current_a),
                            "question_words": len(current_q.split()),
                            "answer_words": len(current_a.split()),
                            "source_type": self.__class__.__name__.lower()
                        }
                    })
                    current_q, current_a = None, None
                else:
                    parsing_stats['incomplete_pairs'] += 1
            elif line and not line.startswith("Q:") and not line.startswith("A:"):
                parsing_stats['unparseable_lines'] += 1
        
        # Check for final incomplete pair
        if current_q and not current_a:
            parsing_stats['incomplete_pairs'] += 1
        
        return qa_pairs, parsing_stats
    
    def calculate_resource_metrics(self, start_time: float, input_file: str) -> Dict[str, Any]:
        """Calculate memory, disk, and resource utilization metrics."""
        # Memory metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # GPU memory metrics
        gpu_memory = {}
        if self.use_gpu and torch.cuda.is_available():
            gpu_memory = {
                'current_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'current_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                'gpu_device_name': torch.cuda.get_device_name(),
                'gpu_capability': torch.cuda.get_device_capability(),
                'total_gpu_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
        
        # Disk I/O metrics
        input_size = os.path.getsize(input_file) if os.path.exists(input_file) else 0
        
        return {
            'system_resources': {
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / 1024**2,
                'memory_vms_mb': memory_info.vms / 1024**2,
                'num_cpu_cores': psutil.cpu_count(),
                'available_memory_gb': psutil.virtual_memory().available / 1024**3
            },
            'gpu_resources': gpu_memory,
            'disk_io': {
                'input_file_size_mb': input_size / 1024**2,
                'input_file_size_bytes': input_size
            }
        }
    
    def calculate_token_metrics(self, prompts: List[str], responses: List[str], 
                              inference_time: float) -> Dict[str, Any]:
        """Calculate token-level performance metrics."""
        prompt_tokens = []
        response_tokens = []
        truncation_count = 0
        
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, truncation=True, max_length=8192)
            prompt_tokens.append(len(tokens))
            if len(self.tokenizer.encode(prompt, truncation=False)) > 8192:
                truncation_count += 1
        
        for response in responses:
            tokens = self.tokenizer.encode(response)
            response_tokens.append(len(tokens))
        
        total_prompt_tokens = sum(prompt_tokens)
        total_response_tokens = sum(response_tokens)
        total_tokens = total_prompt_tokens + total_response_tokens
        
        return {
            'token_statistics': {
                'total_input_tokens': total_prompt_tokens,
                'total_output_tokens': total_response_tokens,
                'total_tokens_processed': total_tokens,
                'avg_prompt_tokens': statistics.mean(prompt_tokens),
                'avg_response_tokens': statistics.mean(response_tokens),
                'max_prompt_tokens': max(prompt_tokens),
                'max_response_tokens': max(response_tokens),
                'truncation_rate': truncation_count / len(prompts) if prompts else 0
            },
            'throughput_metrics': {
                'tokens_per_second': total_tokens / inference_time if inference_time > 0 else 0,
                'input_tokens_per_second': total_prompt_tokens / inference_time if inference_time > 0 else 0,
                'output_tokens_per_second': total_response_tokens / inference_time if inference_time > 0 else 0,
                'effective_context_utilization': statistics.mean(prompt_tokens) / 8192
            }
        }
    
    def save_results(self, qa_pairs: List[Dict], performance_metrics: Dict, 
                    output_file: str):
        """Save Q&A pairs and performance metrics."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save performance report
        report_file = output_path.with_suffix('.performance.json')
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            def write_results():
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pair in qa_pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
            def write_performance_report():
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
            
            # Write both files in parallel
            results_future = executor.submit(write_results)
            report_future = executor.submit(write_performance_report)
            
            results_future.result()
            report_future.result()
        
        return str(report_file)
    
    @abstractmethod
    def process_file(self, input_file: str, output_file: str, **kwargs):
        """Process a file and extract/generate Q&A pairs."""
        pass