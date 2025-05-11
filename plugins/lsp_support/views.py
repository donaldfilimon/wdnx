from flask import Blueprint, jsonify, request

lsp_bp = Blueprint("lsp", __name__, url_prefix="/api/lsp")


@lsp_bp.route("/complete", methods=["POST"])
def complete():
    """Stub code completion endpoint."""
    data = request.get_json(force=True) or {}
    data.get("code", "")
    data.get("position", {})
    # TODO: integrate with LSP server
    return jsonify({"suggestions": []})


@lsp_bp.route("/hover", methods=["POST"])
def hover():
    """Stub hover information endpoint."""
    request.get_json(force=True) or {}
    # TODO: integrate with LSP server
    return jsonify({"hover": {}})


@lsp_bp.route("/definitions", methods=["POST"])
def definitions():
    """Stub go-to-definition endpoint."""
    request.get_json(force=True) or {}
    # TODO: integrate with LSP server
    return jsonify({"definitions": []})
