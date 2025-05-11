"""
embeddings.py - Handler for text embeddings and similarity using OpenAI.
"""

import os
import openai
from typing import List
import numpy as np

# Initialize OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingHandler:
    """
    Handler to create and compare text embeddings via OpenAI.
    """
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts. Returns list of vectors.
        """
        response = openai.Embedding.create(model=self.model, input=texts)
        return [data_point["embedding"] for data_point in response.get("data", [])]

    @staticmethod
    def similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) 