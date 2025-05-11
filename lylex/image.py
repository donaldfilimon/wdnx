"""
image.py - Handler to generate images from text prompts using OpenAI DALL·E.
"""

import os
from typing import List

import openai

# Initialize OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


class ImageGenerator:
    """
    Handler for generating images via OpenAI DALL·E.
    """

    def __init__(self, model: str = "dall-e-3", size: str = "512x512", n: int = 1):
        self.model = model
        self.size = size
        self.n = n

    def generate(self, prompt: str) -> List[str]:
        """
        Generate images for the given prompt. Returns a list of image URLs.
        """
        response = openai.Image.create(prompt=prompt, n=self.n, size=self.size, model=self.model)
        # Extract URLs from response
        return [item.get("url") for item in response.get("data", [])]
