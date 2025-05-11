from flask import Blueprint, request, jsonify
from .utils import scrape_url
from plugin_utils.validation import require_query_params, require_json_fields
from plugin_utils.metrics import metrics

scraper_bp = Blueprint("web_scraper", __name__, url_prefix="/scrape")

@scraper_bp.route("", methods=["GET"])
@require_query_params('url')
@metrics
def scrape_get():
    """Scrape a URL provided via query string."""
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' parameter."}), 400
    try:
        data = scrape_url(url)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@scraper_bp.route("", methods=["POST"])
@require_json_fields('url')
@metrics
def scrape_post():
    """Scrape a URL provided via JSON body."""
    json_data = request.get_json(silent=True) or {}
    url = json_data.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' field in JSON body."}), 400
    try:
        data = scrape_url(url)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500 