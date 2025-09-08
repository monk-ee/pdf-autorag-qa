"""
Q&A Extraction Library

A common library for extracting and generating Q&A pairs from various sources
using Hugging Face transformers with GPU acceleration.
"""

from .base_extractor import BaseQAExtractor
from .transcript_extractor import TranscriptQAExtractor  
from .pdf_generator import PDFQAGenerator
from .performance_analytics import PerformanceAnalyzer
from .text_processing import TextProcessor
from .prompt_manager import PromptManager

__version__ = "1.0.0"
__all__ = [
    "BaseQAExtractor",
    "TranscriptQAExtractor", 
    "PDFQAGenerator",
    "PerformanceAnalyzer",
    "TextProcessor",
    "PromptManager"
]