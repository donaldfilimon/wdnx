"""
vision.py - Handler for large vision models (LVMs) using HuggingFace CLIP.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

class VisionModelHandler:
    """
    Singleton handler to load and run vision models for image features and zero-shot classification.
    """
    _instance = None

    def __new__(cls, model_name: str = "openai/clip-vit-base-patch32"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self._initialized = True

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract raw image feature embeddings.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features

    def classify(self, image: Image.Image, text_labels: List[str]) -> Dict[str, float]:
        """
        Zero-shot classification: returns a mapping label -> probability.
        """
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        return {label: float(probs[0, idx]) for idx, label in enumerate(text_labels)} 