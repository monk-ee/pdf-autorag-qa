"""
Text Processing Utilities

Common text processing functions for VTT, PDF, and other formats.
"""
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for text processing and cleaning."""
    
    @staticmethod
    def clean_vtt_text(content: str) -> str:
        """Clean and normalize VTT content with advanced reconstruction."""
        # Step 1: Remove timing tags that break words BEFORE line processing
        content = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>', '', content)
        content = re.sub(r'</c>', '', content)
        content = re.sub(r'<c>', '', content)
        
        # Step 2: Clean HTML entities
        content = content.replace('&gt;&gt;', '>>')
        content = content.replace('&lt;', '<')
        content = content.replace('&amp;', '&')
        
        return content
    
    @staticmethod
    def extract_vtt_segments(content: str) -> List[str]:
        """Extract text segments from VTT content."""
        lines = content.splitlines()
        text_segments = []
        seen_segments = set()
        
        for line in lines:
            line = line.strip()
            
            # Skip VTT headers and metadata
            if (line.startswith('WEBVTT') or line.startswith('Kind:') or 
                line.startswith('Language:') or line.startswith('align:') or 
                line.startswith('position:')):
                continue
            
            # Skip timestamp lines
            if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} -->", line):
                continue
            
            # Skip empty lines and standalone numbers
            if line == '' or re.match(r"^\d+$", line):
                continue
            
            # Skip duplicate segments (VTT repeats captions)
            if line and line not in seen_segments:
                seen_segments.add(line)
                text_segments.append(line)
        
        return text_segments
    
    @staticmethod
    def reconstruct_sentences(segments: List[str]) -> str:
        """Reconstruct proper sentences from fragmented segments."""
        if not segments:
            return ""
        
        # Patterns for incomplete sentence endings that should continue
        continuation_patterns = r'\b(and|or|but|with|from|to|in|on|at|the|a|an|of|for|by|as|is|was|were|are|have|has|had|will|would|could|should|may|might|can|do|does|did|be|been|being|this|that|these|those|his|her|its|their|our|my|your)\.?\s*$'
        
        # Patterns for single words that are likely fragments
        single_word_pattern = r'^[a-zA-Z]+[.,!?]?$'
        
        reconstructed = []
        current_sentence = ""
        
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            
            # Check if this segment should continue the previous sentence
            should_continue = False
            
            if current_sentence:
                # Continue if previous segment ended with continuation word
                if re.search(continuation_patterns, current_sentence, re.IGNORECASE):
                    should_continue = True
                # Continue if current segment is a single word fragment
                elif re.match(single_word_pattern, segment) and len(segment) < 15:
                    should_continue = True
                # Continue if current segment doesn't start with capital or speaker marker
                elif not segment[0].isupper() and not segment.startswith('>>'):
                    should_continue = True
            
            if should_continue:
                # Add space if current sentence doesn't end with punctuation
                if current_sentence and not current_sentence[-1] in '.,!?':
                    current_sentence += " " + segment
                else:
                    current_sentence += " " + segment
            else:
                # Start new sentence
                if current_sentence:
                    reconstructed.append(current_sentence.strip())
                current_sentence = segment
        
        # Add final sentence
        if current_sentence:
            reconstructed.append(current_sentence.strip())
        
        # Final cleanup and capitalization fixes
        final_text = " ".join(reconstructed)
        
        # Fix capitalization after periods
        final_text = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), final_text)
        
        # Fix multiple spaces
        final_text = re.sub(r'\s+', ' ', final_text)
        
        # Ensure speaker markers are properly formatted
        final_text = re.sub(r'>>\s*([a-z])', lambda m: '>> ' + m.group(1).upper(), final_text)
        
        return final_text.strip()
    
    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """Clean and normalize PDF text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)  # Join broken sentences
        
        # Clean up page numbers and headers/footers (basic heuristics)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers
            if re.match(r'^\d+$', line) and len(line) < 4:
                continue
            # Skip very short lines that are likely headers/footers
            if len(line) < 10 and not re.match(r'^[A-Z]', line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def analyze_text_structure(blocks_dict: dict, page_num: int) -> List[Dict]:
        """Analyze text structure to identify sections, headings, etc."""
        sections = []
        
        if 'blocks' not in blocks_dict:
            return sections
        
        for block in blocks_dict['blocks']:
            if 'lines' not in block:
                continue
                
            for line in block['lines']:
                if 'spans' not in line:
                    continue
                    
                for span in line['spans']:
                    text = span.get('text', '').strip()
                    if not text:
                        continue
                    
                    # Analyze font properties to identify headings
                    font_size = span.get('size', 0)
                    font_flags = span.get('flags', 0)
                    is_bold = bool(font_flags & 2**4)
                    
                    # Heuristics for section identification
                    is_heading = (
                        font_size > 12 or 
                        is_bold or
                        (len(text) < 100 and text.isupper()) or
                        re.match(r'^\d+\.?\s+[A-Z]', text)
                    )
                    
                    if is_heading:
                        sections.append({
                            'page': page_num,
                            'text': text,
                            'font_size': font_size,
                            'is_bold': is_bold,
                            'type': 'heading' if font_size > 14 or text.isupper() else 'subheading'
                        })
        
        return sections
    
    @staticmethod
    def intelligent_chunking(text: str, chunk_size: int, overlap: int, 
                           structure_info: List[Dict] = None) -> List[Dict]:
        """Create intelligent chunks preserving content structure."""
        words = text.split()
        chunks = []
        
        # Basic sliding window chunking (can be enhanced with structure info)
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_word': i,
                'end_word': min(i + chunk_size, len(words)),
                'word_count': len(chunk_words),
                'type': 'intelligent' if structure_info else 'sliding_window'
            })
        
        return chunks