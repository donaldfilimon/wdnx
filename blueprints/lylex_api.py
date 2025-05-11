"""
Lylex API Blueprint

Defines the Flask blueprint exposing endpoints for managing Lylex interactions, shards, chat, drift detection, personas, content filtering, and backend information.
"""

from flask import Blueprint, jsonify, request
from app import lylex_db
from lylex.ai import LylexAgent

lylex_api = Blueprint("lylex_api", __name__, url_prefix="/lylex")


@lylex_api.route("/interactions/<int:interaction_id>", methods=["DELETE"])
def delete_interaction(interaction_id):
    """Delete a stored interaction by ID"""
    success = lylex_db.delete_interaction(interaction_id)
    return jsonify({"deleted": success})


@lylex_api.route("/interactions/<int:interaction_id>", methods=["PUT"])
def update_interaction_metadata(interaction_id):
    """Update metadata for a stored interaction"""
    data = request.get_json(force=True) or {}
    success = lylex_db.update_interaction_metadata(
        interaction_id, data.get("metadata", {})
    )
    return jsonify({"updated": success})


@lylex_api.route("/interactions/<int:interaction_id>", methods=["GET"])
def get_interaction_metadata(interaction_id):
    """Retrieve metadata for a stored interaction"""
    metadata = lylex_db.get_interaction_metadata(interaction_id)
    return jsonify({"metadata": metadata})


@lylex_api.route("/interactions/count", methods=["GET"])
def count_interactions():
    """Return total number of interactions stored"""
    count = lylex_db.count_interactions()
    return jsonify({"count": count})


@lylex_api.route("/health/ping", methods=["GET"])
def ping_backend():
    """Ping the Lylex backend for health check"""
    alive = lylex_db.ping_backend()
    return jsonify({"alive": alive})


@lylex_api.route("/interactions/flush", methods=["DELETE"])
def flush_interactions():
    """Remove all stored interactions"""
    lylex_db.flush_interactions()
    return jsonify({"flushed": True})


@lylex_api.route("/interactions/export", methods=["GET"])
def export_interactions():
    """Export all stored interactions as JSON list"""
    try:
        data = lylex_db.export_interactions()
        return jsonify({"interactions": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/shards", methods=["GET"])
def list_shards():
    """List configured shard nodes for Lylex."""
    try:
        shards = (
            list(lylex_db.client.shards.keys())
            if hasattr(lylex_db.client, "shards")
            else []
        )
        return jsonify({"shards": shards})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/shards/health", methods=["GET"])
def shards_health():
    """Get health status of each Lylex shard."""
    try:
        health = (
            lylex_db.client.check_shards_health()
            if hasattr(lylex_db.client, "check_shards_health")
            else {}
        )
        return jsonify({"health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/chat", methods=["POST"])
def api_chat():
    """Interactive chat endpoint powered by LylexAgent"""
    payload = request.get_json(force=True) or {}
    prompt = payload.get("prompt", "")
    model_name = payload.get("model_name", "gpt2")
    backend = payload.get("backend", "pt")
    memory_limit = payload.get("memory_limit", 5)
    try:
        agent = LylexAgent(
            model_name=model_name,
            backend=backend,
            memory_db=lylex_db,
            memory_limit=memory_limit,
        )
        response = agent.chat(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/drift", methods=["POST"])
def detect_drift():
    """Detect drift in vector space with optional threshold"""
    data = request.get_json(force=True) or {}
    threshold = data.get("threshold", 0.1)
    try:
        drifted = lylex_db.client.detect_drift(threshold)
        return jsonify({"drift_detected": drifted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/personas", methods=["GET"])
def list_personas():
    """List available personas"""
    try:
        personas = lylex_db.client.list_personas()
        return jsonify({"personas": personas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/personas", methods=["POST"])
def create_persona():
    """Create a new persona"""
    data = request.get_json(force=True) or {}
    name = data.get("name")
    config = data.get("config", {})
    if not name:
        return jsonify({"error": "Missing persona name"}), 400
    try:
        pid = lylex_db.client.create_persona(name, config)
        return jsonify({"persona_id": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/personas/switch", methods=["POST"])
def switch_persona():
    """Switch to a different persona"""
    data = request.get_json(force=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing persona name"}), 400
    try:
        lylex_db.client.switch_persona(name)
        return jsonify({"switched": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/filter", methods=["POST"])
def api_filter_content():
    """Filter a text for safety or compliance"""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    try:
        result = lylex_db.client.filter_content(text)
        return jsonify({"filtered": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/mitigate", methods=["POST"])
def api_mitigate_bias():
    """Mitigate bias in the given text"""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    try:
        result = lylex_db.client.mitigate_bias(text)
        return jsonify({"mitigated": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/version", methods=["GET"])
def backend_version():
    """Get underlying backend version"""
    try:
        v = lylex_db.client.get_backend_version()
        return jsonify({"version": v})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/features/<feature>", methods=["GET"])
def supports_feature(feature):
    """Check if backend supports a specific feature"""
    try:
        ok = lylex_db.client.supports_feature(feature)
        return jsonify({"supported": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/blocks", methods=["GET"])
def list_blocks():
    """List all blockchain blocks"""
    try:
        blocks = lylex_db.client.list_blocks()
        return jsonify({"blocks": blocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/blocks/<int:block_id>", methods=["GET"])
def get_block(block_id):
    """Get a specific block detail"""
    try:
        blk = lylex_db.client.get_block(block_id)
        return jsonify({"block": blk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/transactions", methods=["GET"])
def list_transactions():
    """List all MVCC transactions"""
    try:
        txs = lylex_db.client.list_transactions()
        return jsonify({"transactions": txs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lylex_api.route("/transactions/<int:tx_id>", methods=["GET"])
def get_transaction(tx_id):
    """Get details of a MVCC transaction"""
    try:
        tx = lylex_db.client.get_transaction(tx_id)
        return jsonify({"transaction": tx})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
