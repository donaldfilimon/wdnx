import os

import requests
from flask import Blueprint, jsonify, request

from plugin_utils.logging import get_logger
from plugin_utils.metrics import metrics
from plugin_utils.validation import require_json_fields
from app import settings

ollama_bp = Blueprint("ollama_provider", __name__, url_prefix="/ollama")
logger = get_logger(__name__)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")


@ollama_bp.route("/generate", methods=["POST"])
@require_json_fields("model", "prompt")
@metrics
def generate_ollama():
    """
    Proxy endpoint for Ollama /api/generate.
    Expects 'model', 'prompt', and optionally 'stream', 'options'.
    """
    data = request.get_json(force=True)
    # Validation for model and prompt handled by decorator
    logger.info(f"Ollama generate request for model: {data.get('model')}")
    try:
        response = requests.post(
            OLLAMA_API_URL, json=data, stream=data.get("stream", False)
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        if data.get("stream", False):
            # For streaming responses, we'd ideally return a StreamingHttpResponse
            # but for simplicity, we'll just return the first chunk or an indicator.
            # A real implementation would require more robust handling.
            # For now, just indicate streaming is not fully supported here this way.
            logger.info(
                f"Ollama streaming response initiated for model: {data.get('model')}"
            )
            return jsonify(
                {
                    "message": "Streaming initiated, but this proxy doesn't fully support it. Check Ollama directly."
                }
            )
        else:
            logger.info(
                f"Ollama non-streaming response successful for model: {data.get('model')}"
            )
            return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Ollama request failed for model {data.get('model')}: {e}", exc_info=True
        )
        raise  # Reraise to be caught by centralized error handler, will produce 502 or similar
    except Exception as e:
        logger.error(
            f"Ollama generic error for model {data.get('model')}: {e}", exc_info=True
        )
        raise


@ollama_bp.route("/models", methods=["GET"])
def list_models():
    """List available Ollama models."""
    url = settings.ollama_api_url.rstrip("/") + "/api/v1/models"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ollama_bp.route("/complete", methods=["POST"])
def generate_completion():
    """Generate completion using Ollama model."""
    data = request.get_json(force=True) or {}
    model = data.get("model")
    prompt = data.get("prompt")
    params = data.get("params", {})
    if not model or not prompt:
        return jsonify({"error": "Missing 'model' or 'prompt'"}), 400
    url = settings.ollama_api_url.rstrip("/") + "/api/v1/completions"
    payload = {"model": model, "prompt": prompt, **params}
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
