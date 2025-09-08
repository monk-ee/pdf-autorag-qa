"""
Prompt Manager for Q&A Generation

Handles loading and formatting prompts from configuration files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompts for different Q&A generation tasks."""
    
    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.prompts = {}
        self.load_prompts()
    
    def load_prompts(self):
        """Load prompts from JSON configuration file."""
        try:
            if self.prompts_file.exists():
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    self.prompts = json.load(f)
                logger.info(f"Loaded prompts from {self.prompts_file}")
            else:
                logger.warning(f"Prompts file not found: {self.prompts_file}")
                self._create_default_prompts()
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self._create_default_prompts()
    
    def _create_default_prompts(self):
        """Create default prompts if file doesn't exist."""
        self.prompts = {
            "pdf_qa_generation": {
                "instruct": {
                    "system_prefix": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
                    "system_suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    "templates": {
                        "basic": "Generate basic educational Q&A pairs from this content.\n\nContent:\n{chunk_text}\n\nGenerate 2-4 Q&A pairs:",
                        "intermediate": "Generate intermediate educational Q&A pairs from this content.\n\nContent:\n{chunk_text}\n\nGenerate 2-4 Q&A pairs:",
                        "advanced": "Generate advanced educational Q&A pairs from this content.\n\nContent:\n{chunk_text}\n\nGenerate 2-4 Q&A pairs:"
                    }
                },
                "non_instruct": {
                    "system_prefix": "",
                    "system_suffix": "",
                    "templates": {
                        "basic": "Generate basic Q&A pairs from this content:\n\n{chunk_text}\n\nQ&A Pairs:",
                        "intermediate": "Generate intermediate Q&A pairs from this content:\n\n{chunk_text}\n\nQ&A Pairs:",
                        "advanced": "Generate advanced Q&A pairs from this content:\n\n{chunk_text}\n\nQ&A Pairs:"
                    }
                }
            }
        }
        logger.info("Using default prompts")
    
    def is_instruct_model(self, model_name: str) -> bool:
        """Determine if model is an instruct model based on name."""
        instruct_indicators = ["instruct", "chat", "it", "sft"]
        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in instruct_indicators)
    
    def get_prompt_template(self, task: str, model_name: str, 
                          difficulty: str = "basic", custom_template: Optional[str] = None) -> Dict[str, str]:
        """Get prompt template for a specific task and model type."""
        if custom_template:
            # Use custom template if provided
            is_instruct = self.is_instruct_model(model_name)
            template_type = "instruct" if is_instruct else "non_instruct"
            
            # Get system formatting from default task
            default_task = list(self.prompts.keys())[0] if self.prompts else "pdf_qa_generation"
            system_config = self.prompts.get(default_task, {}).get(template_type, {})
            
            return {
                "system_prefix": system_config.get("system_prefix", ""),
                "system_suffix": system_config.get("system_suffix", ""),
                "template": custom_template
            }
        
        # Use configured templates
        if task not in self.prompts:
            logger.warning(f"Task '{task}' not found in prompts, using default")
            task = list(self.prompts.keys())[0] if self.prompts else "pdf_qa_generation"
        
        task_prompts = self.prompts[task]
        is_instruct = self.is_instruct_model(model_name)
        template_type = "instruct" if is_instruct else "non_instruct"
        
        if template_type not in task_prompts:
            logger.warning(f"Template type '{template_type}' not found, using fallback")
            template_type = "instruct" if "instruct" in task_prompts else list(task_prompts.keys())[0]
        
        config = task_prompts[template_type]
        templates = config.get("templates", {})
        
        if difficulty not in templates:
            logger.warning(f"Difficulty '{difficulty}' not found, using available template")
            difficulty = list(templates.keys())[0] if templates else "basic"
        
        return {
            "system_prefix": config.get("system_prefix", ""),
            "system_suffix": config.get("system_suffix", ""),
            "template": templates[difficulty]
        }
    
    def format_prompt(self, task: str, model_name: str, chunk_text: str,
                     difficulty: str = "basic", custom_template: Optional[str] = None,
                     **kwargs) -> str:
        """Format a complete prompt with the given parameters."""
        prompt_config = self.get_prompt_template(task, model_name, difficulty, custom_template)
        
        # Format the template with provided variables
        template_vars = {
            "chunk_text": chunk_text,
            **kwargs
        }
        
        try:
            formatted_template = prompt_config["template"].format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using raw template")
            formatted_template = prompt_config["template"]
        
        # Combine with system formatting
        full_prompt = (
            prompt_config["system_prefix"] + 
            formatted_template + 
            prompt_config["system_suffix"]
        )
        
        return full_prompt
    
    def create_prompts_for_chunks(self, task: str, model_name: str, chunks: List[Any],
                                 difficulty_levels: List[str] = None,
                                 custom_template: Optional[str] = None,
                                 **kwargs) -> List[str]:
        """Create prompts for multiple chunks with rotating difficulty levels."""
        if difficulty_levels is None:
            difficulty_levels = ['basic', 'intermediate', 'advanced']
        
        prompts = []
        
        for i, chunk in enumerate(chunks):
            # Rotate through difficulty levels
            difficulty = difficulty_levels[i % len(difficulty_levels)]
            
            # Extract text from chunk (handle both dict and string formats)
            chunk_text = chunk['text'] if isinstance(chunk, dict) else str(chunk)
            
            prompt = self.format_prompt(
                task=task,
                model_name=model_name,
                chunk_text=chunk_text,
                difficulty=difficulty,
                custom_template=custom_template,
                **kwargs
            )
            
            prompts.append(prompt)
        
        return prompts
    
    def add_custom_prompt(self, task: str, template_type: str, difficulty: str, 
                         template: str, system_prefix: str = "", system_suffix: str = ""):
        """Add a custom prompt template at runtime."""
        if task not in self.prompts:
            self.prompts[task] = {}
        
        if template_type not in self.prompts[task]:
            self.prompts[task][template_type] = {
                "system_prefix": system_prefix,
                "system_suffix": system_suffix,
                "templates": {}
            }
        
        self.prompts[task][template_type]["templates"][difficulty] = template
        logger.info(f"Added custom prompt: {task}/{template_type}/{difficulty}")
    
    def save_prompts(self, output_file: Optional[str] = None):
        """Save current prompts to file."""
        output_path = Path(output_file) if output_file else self.prompts_file
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved prompts to {output_path}")
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
    
    def list_available_prompts(self) -> Dict[str, Any]:
        """Get a summary of available prompts."""
        summary = {}
        for task, task_config in self.prompts.items():
            summary[task] = {}
            for template_type, type_config in task_config.items():
                templates = type_config.get("templates", {})
                summary[task][template_type] = list(templates.keys())
        
        return summary
    
    def validate_prompts(self) -> Dict[str, List[str]]:
        """Validate prompt configuration and return any issues."""
        issues = {"errors": [], "warnings": []}
        
        for task, task_config in self.prompts.items():
            if not isinstance(task_config, dict):
                issues["errors"].append(f"Task '{task}' must be a dictionary")
                continue
            
            for template_type in ["instruct", "non_instruct"]:
                if template_type not in task_config:
                    issues["warnings"].append(f"Task '{task}' missing '{template_type}' templates")
                    continue
                
                type_config = task_config[template_type]
                if "templates" not in type_config:
                    issues["errors"].append(f"Task '{task}' {template_type} missing 'templates' section")
                    continue
                
                templates = type_config["templates"]
                if not templates:
                    issues["warnings"].append(f"Task '{task}' {template_type} has no templates")
                
                # Check for required template variables
                for difficulty, template in templates.items():
                    if "{chunk_text}" not in template:
                        issues["warnings"].append(
                            f"Template {task}/{template_type}/{difficulty} missing {{chunk_text}} placeholder"
                        )
        
        return issues