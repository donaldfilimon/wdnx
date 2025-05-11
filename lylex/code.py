"""
code.py - Handler for large code models (LCMs) using HuggingFace.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeModelHandler:
    """
    Singleton handler to load and run large code models for code generation.
    """
    _instances = {}

    def __new__(cls, model_name: str, device: str = None):
        key = model_name
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, model_name: str, device: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = device or 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self._initialized = True

    def generate_code(self, prompt: str, max_length: int = 256, **kwargs) -> str:
        """
        Generate code completion given a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 