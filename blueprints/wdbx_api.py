"""
WDBX API Blueprint

Defines the Flask blueprint exposing WDBX backend endpoints for vector operations, artifacts, self-updates, shards, API keys, audit logs, and advanced features.
"""

import json

from flask import Blueprint, Response, jsonify, request, send_file

from app import wdbx

wdbx_api = Blueprint("wdbx_api", __name__, url_prefix="/wdbx")


# Vector endpoints
@wdbx_api.route("/vector/store", methods=["POST"])
def api_store_vector() -> Response:
    """
    Store a single vector in the vector store.

    Args:
        vector (list[float|int]): The embedding vector to store (from JSON body under "vector").
        metadata (dict[str, Any], optional): Associated metadata tags (from JSON body under "metadata").

    Returns:
        Response: JSON response containing:
            - vector_id (int): The ID of the stored vector on success.
            - error (str): Error message on failure (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    try:
        vid = wdbx.store(data.get("vector", []), data.get("metadata", {}))
        return jsonify({"vector_id": vid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/vector/search", methods=["POST"])
def api_search_vector() -> Response:
    """
    Search for similar vectors in the store.

    Args:
        vector (list[float|int]): The query vector (from JSON body under "vector").
        limit (int, optional): Maximum number of results to return (from JSON body under "limit", default 10).

    Returns:
        Response: JSON response containing:
            - results (list[dict]): Each result has keys "id", "score", and "metadata".
            - error (str): Error message on failure (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    try:
        raw = wdbx.search(data.get("vector", []), limit=data.get("limit", 10))
        results = [{"id": r[0], "score": r[1], "metadata": r[2]} for r in raw]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/vector/bulk_store", methods=["POST"])
def api_bulk_store_vectors() -> Response:
    """
    Bulk store multiple vectors with metadata.

    Args:
        items (list[dict]): List of items, each with:
            - vector (list[float|int]): The vector embedding.
            - metadata (dict): Associated metadata.

    Returns:
        Response: JSON containing {"vector_ids": list[int]} on success or {"error": str} on failure.

    Raises:
        Exception: If storage fails (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    pairs = [(it.get("vector", []), it.get("metadata", {})) for it in data.get("items", [])]
    try:
        vids = wdbx.bulk_store(pairs)
        return jsonify({"vector_ids": vids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/vector/bulk_search", methods=["POST"])
def api_bulk_search_vectors() -> Response:
    """
    Bulk search for multiple query vectors.

    Args:
        vectors (list[list[float|int]]): List of query vectors.
        limit (int, optional): Max results per query (default 10).

    Returns:
        Response: JSON containing {"results": list[list[dict]]} or {"error": str}.

    Raises:
        Exception: If search fails (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    try:
        results = wdbx.bulk_search(data.get("vectors", []), limit=data.get("limit", 10))
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Artifact endpoints
@wdbx_api.route("/artifact/store", methods=["POST"])
def api_store_artifact() -> Response:
    """
    Store an artifact (e.g., model file) uploaded as form-data.

    Args:
        file (FileStorage): The file under form key "file".
        metadata (dict, optional): JSON metadata under form key "metadata".

    Returns:
        Response: JSON containing {"artifact_id": int} on success or {"error": str} on failure.

    Raises:
        400: If no file provided.
        Exception: If storage fails (HTTP 500).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    file.stream
    meta = {}
    try:
        meta = request.form.get("metadata") and json.loads(request.form["metadata"]) or {}
    except Exception:
        pass
    try:
        aid = wdbx.store_model(file, meta) if hasattr(wdbx, "store_model") else None
        return jsonify({"artifact_id": aid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/artifact/load/<int:artifact_id>", methods=["GET"])
def api_load_artifact(artifact_id: int) -> Response:
    """
    Load/download a stored artifact by ID.

    Args:
        artifact_id (int): The ID of the artifact to download.

    Returns:
        Response: File download attachment on success or JSON {"error": str} on failure.

    Raises:
        Exception: If loading fails (HTTP 500).
    """
    try:
        tmp_path = wdbx.load_model(artifact_id, None)
        return send_file(tmp_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Self-update endpoints for WDBX
@wdbx_api.route("/ai_update", methods=["POST"])
def api_wdbx_ai_update() -> Response:
    """
    Perform AI-driven self-update on a file.

    Args:
        file_path (str): Path of the file to update (from JSON body under "file_path").
        instruction (str): Instruction for patch generation (from JSON body under "instruction").
        model_name (str, optional): Name of the AI model to use (from JSON body under "model_name").
        backend (str, optional): Backend to use for AI processing (from JSON body under "backend").
        memory_limit (int, optional): Maximum memory context size for the agent (from JSON body under "memory_limit").

    Returns:
        Response: JSON {"status": "ok"} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    try:
        wdbx.ai_update(
            data["file_path"],
            data["instruction"],
            model_name=data.get("model_name"),
            backend=data.get("backend"),
            memory_limit=data.get("memory_limit"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/git_update", methods=["POST"])
def api_wdbx_git_update() -> Response:
    """
    Perform Git-based self-update on the local repository.

    Args:
        local_dir (str): Local directory path to update (from JSON body under "local_dir").
        module_paths (list[str], optional): Specific module paths to update (from JSON body under "module_paths").

    Returns:
        Response: JSON {"status": "ok"} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    try:
        wdbx.git_update(data["local_dir"], module_paths=data.get("module_paths"))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/schedule_self_update", methods=["POST"])
def api_wdbx_schedule_self_update() -> Response:
    """
    Schedule periodic self-update tasks via Git.

    Args:
        interval (int): Update interval in seconds (from JSON body under "interval").
        repo_dir (str): Path to the repository directory (from JSON body under "repo_dir").
        module_paths (list[str], optional): Module paths to include in the update (from JSON body under "module_paths").

    Returns:
        Response: JSON {"status": "ok"} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    try:
        wdbx.schedule_self_update(
            interval=data["interval"],
            repo_dir=data["repo_dir"],
            module_paths=data.get("module_paths"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/stop_self_update", methods=["POST"])
def api_wdbx_stop_self_update() -> Response:
    """
    Stop any scheduled self-update jobs.

    Returns:
        Response: JSON {"status": "ok"} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        wdbx.stop_self_update()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/rollback_update", methods=["POST"])
def api_wdbx_rollback_update() -> Response:
    """
    Roll back a file to a previous version using backups.

    Args:
        file_path (str): Path of the file to restore (from JSON body under "file_path").
        backup_file (str, optional): Specific backup file to use (from JSON body under "backup_file").

    Returns:
        Response: JSON {"status": "ok"} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    try:
        wdbx.rollback_update(data["file_path"], backup_file=data.get("backup_file"))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/shards", methods=["GET"])
def api_list_shards() -> Response:
    """
    List configured shard nodes.

    Returns:
        Response: JSON {"shards": list[str]} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        shards = list(wdbx.shards.keys()) if hasattr(wdbx, "shards") else []
        return jsonify({"shards": shards})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/shards/health", methods=["GET"])
def api_shards_health() -> Response:
    """
    Get health status of each shard node.

    Returns:
        Response: JSON {"health": dict} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        health = wdbx.check_shards_health() if hasattr(wdbx, "check_shards_health") else {}
        return jsonify({"health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- API Key Management ---
@wdbx_api.route("/apikey/generate", methods=["POST"])
def api_generate_api_key() -> Response:
    """
    Generate a new API key for a user.

    Args:
        user_id (str): Identifier of the user (from JSON body under "user_id").

    Returns:
        Response: JSON {"api_key": str} on success, HTTP 400 if user_id missing, or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        key = wdbx.generate_api_key(user_id)
        return jsonify({"api_key": key})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/apikey/revoke", methods=["POST"])
def api_revoke_api_key() -> Response:
    """
    Revoke an existing API key.

    Args:
        key (str): The API key to revoke (from JSON body under "key").

    Returns:
        Response: JSON {"revoked": bool} on success, HTTP 400 if key missing, or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    key = data.get("key")
    if not key:
        return jsonify({"error": "Missing key"}), 400
    try:
        success = wdbx.revoke_api_key(key)
        return jsonify({"revoked": success})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Audit Logs ---
@wdbx_api.route("/audit_logs", methods=["GET"])
def api_get_audit_logs() -> Response:
    """
    Retrieve audit logs optionally filtered by a timestamp.

    Args:
        since (str, optional): ISO timestamp to filter logs (from query param "since").

    Returns:
        Response: JSON {"logs": list} on success or {"error": str} with HTTP 500 on failure.
    """
    since = request.args.get("since")
    try:
        logs = wdbx.get_audit_logs(since)
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Advanced Features ---
@wdbx_api.route("/drift", methods=["POST"])
def api_detect_drift() -> Response:
    """
    Detect drift in the vector store based on a threshold.

    Args:
        threshold (float, optional): Drift detection threshold (from JSON body under "threshold", default 0.1).

    Returns:
        Response: JSON {"drift_detected": bool} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    threshold = data.get("threshold", 0.1)
    try:
        drifted = wdbx.detect_drift(threshold)
        return jsonify({"drift_detected": drifted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/personas", methods=["GET"])
def api_list_personas() -> Response:
    """
    List configured personas available in the backend.

    Returns:
        Response: JSON {"personas": list} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        personas = wdbx.list_personas()
        return jsonify({"personas": personas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/personas", methods=["POST"])
def api_create_persona() -> Response:
    """
    Create a new persona configuration.

    Args:
        name (str): Name of the persona (from JSON body under "name").
        config (dict, optional): Persona configuration (from JSON body under "config").

    Returns:
        Response: JSON {"persona_id": int} on success, HTTP 400 if name missing, or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    name = data.get("name")
    config = data.get("config", {})
    if not name:
        return jsonify({"error": "Missing persona name"}), 400
    try:
        pid = wdbx.create_persona(name, config)
        return jsonify({"persona_id": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/personas/switch", methods=["POST"])
def api_switch_persona() -> Response:
    """
    Switch the active persona.

    Args:
        name (str): Name of the persona to switch to (from JSON body under "name").

    Returns:
        Response: JSON {"switched": True} on success, HTTP 400 if name missing, or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing persona name"}), 400
    try:
        wdbx.switch_persona(name)
        return jsonify({"switched": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/filter", methods=["POST"])
def api_filter_content() -> Response:
    """
    Filter a text string for safety or compliance.

    Args:
        text (str): The text to filter (from JSON body under "text").

    Returns:
        Response: JSON {"filtered": str} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    try:
        out = wdbx.filter_content(text)
        return jsonify({"filtered": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/mitigate", methods=["POST"])
def api_mitigate_bias() -> Response:
    """
    Mitigate bias in a provided text string.

    Args:
        text (str): The text to mitigate (from JSON body under "text").

    Returns:
        Response: JSON {"mitigated": str} on success or {"error": str} with HTTP 500 on failure.
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    try:
        out = wdbx.mitigate_bias(text)
        return jsonify({"mitigated": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/version", methods=["GET"])
def api_backend_version() -> Response:
    """
    Retrieve the version of the underlying backend.

    Returns:
        Response: JSON {"version": str} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        v = wdbx.get_backend_version()
        return jsonify({"version": v})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/features/<feature>", methods=["GET"])
def api_supports_feature(feature: str) -> Response:
    """
    Check if a specific feature is supported by the backend.

    Args:
        feature (str): Feature name to check (from URL path).

    Returns:
        Response: JSON {"supported": bool} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        ok = wdbx.supports_feature(feature)
        return jsonify({"supported": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/blocks", methods=["GET"])
def api_list_blocks() -> Response:
    """
    List all blockchain blocks.

    Returns:
        Response: JSON {"blocks": list} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        blocks = wdbx.list_blocks()
        return jsonify({"blocks": blocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/blocks/<int:block_id>", methods=["GET"])
def api_get_block(block_id: int) -> Response:
    """
    Retrieve details of a specific blockchain block.

    Args:
        block_id (int): Identifier of the block (from URL path).

    Returns:
        Response: JSON {"block": dict} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        blk = wdbx.get_block(block_id)
        return jsonify({"block": blk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/transactions", methods=["GET"])
def api_list_transactions() -> Response:
    """
    List all MVCC transactions.

    Returns:
        Response: JSON {"transactions": list} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        txs = wdbx.list_transactions()
        return jsonify({"transactions": txs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@wdbx_api.route("/transactions/<int:tx_id>", methods=["GET"])
def api_get_transaction(tx_id: int) -> Response:
    """
    Retrieve details of a specific MVCC transaction.

    Args:
        tx_id (int): Identifier of the transaction (from URL path).

    Returns:
        Response: JSON {"transaction": dict} on success or {"error": str} with HTTP 500 on failure.
    """
    try:
        tx = wdbx.get_transaction(tx_id)
        return jsonify({"transaction": tx})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
