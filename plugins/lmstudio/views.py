from flask import Blueprint, request, jsonify
import openai
from plugin_utils.validation import require_json_fields
from plugin_utils.metrics import metrics
from plugin_utils.logging import get_logger
from config import settings

# Configure OpenAI base URL for LMStudio if not already done globally
# This ensures the plugin uses the correct endpoint if multiple OpenAI-compatible services are used.
if settings.lmstudio_api_url:
    openai.api_base = settings.lmstudio_api_url.rstrip("/")

lmstudio_bp = Blueprint("lmstudio", __name__, url_prefix="/api/lmstudio")
logger = get_logger(__name__)

@lmstudio_bp.route("/models", methods=["GET"])
@metrics
def list_models():
    """List available LMStudio models."""
    logger.info("Request to list LMStudio models.")
    try:
        models = openai.Model.list()
        logger.info(f"Successfully listed {len(models.get('data', []))} LMStudio models.")
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error listing LMStudio models: {e}", exc_info=True)
        raise

@lmstudio_bp.route("/complete", methods=["POST"])
@require_json_fields('model', 'prompt') # Ensure model and prompt are provided
@metrics
def generate_completion():
    """Generate completion using LMStudio model."""
    data = request.get_json(force=True)
    model = data.get("model")
    prompt = data.get("prompt")
    params = data.get("params", {})
    logger.info(f"LMStudio completion request for model: {model}")
    try:
        resp = openai.Completion.create(model=model, prompt=prompt, **params)
        logger.info(f"Successfully generated completion from LMStudio model: {model}")
        return jsonify(resp.to_dict())
    except Exception as e:
        logger.error(f"Error generating completion from LMStudio model {model}: {e}", exc_info=True)
        raise

@lmstudio_bp.route("/embed", methods=["POST"])
def generate_embedding():
    """Generate embeddings using LMStudio model."""
    data = request.get_json(force=True) or {}
    model = data.get("model")
    inputs = data.get("input")
    if not model or inputs is None:
        return jsonify({"error": "Missing 'model' or 'input'"}), 400
    try:
        resp = openai.Embedding.create(model=model, input=inputs)
        return jsonify(resp.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500 