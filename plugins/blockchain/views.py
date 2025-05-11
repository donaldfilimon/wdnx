from flask import Blueprint, render_template, flash, jsonify
from blockchain_utils import list_anchors

blockchain_bp = Blueprint("blockchain", __name__, url_prefix="/blockchain")

@blockchain_bp.route("/anchors", methods=["GET"])
def view_anchors():
    """Render HTML view of anchored hashes."""
    try:
        entries = list_anchors(limit=100)
    except Exception as e:
        flash(f"Failed to retrieve anchors: {e}")
        entries = []
    return render_template("anchors.html", anchors=entries)

@blockchain_bp.route("/api/anchors", methods=["GET"])
def api_anchors():
    """Return JSON list of anchored hashes."""
    try:
        entries = list_anchors(limit=100)
        return jsonify({"anchors": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 