"""
WDBX API Blueprint

Defines the Flask blueprint exposing WDBX backend endpoints for vector operations, artifacts, self-updates, shards, API keys, audit logs, and advanced features.
"""
from flask import Blueprint, request, jsonify, send_file
from app import wdbx
from flask_jwt_extended import verify_jwt_in_request
import json

wdbx_api = Blueprint('wdbx_api', __name__, url_prefix='/wdbx')

# Vector endpoints
@wdbx_api.route('/vector/store', methods=['POST'])
def api_store_vector():
    data = request.get_json(force=True) or {}
    try:
        vid = wdbx.store(data.get('vector', []), data.get('metadata', {}))
        return jsonify({'vector_id': vid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/vector/search', methods=['POST'])
def api_search_vector():
    data = request.get_json(force=True) or {}
    try:
        raw = wdbx.search(data.get('vector', []), limit=data.get('limit', 10))
        results = [{'id': r[0], 'score': r[1], 'metadata': r[2]} for r in raw]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/vector/bulk_store', methods=['POST'])
def api_bulk_store_vectors():
    data = request.get_json(force=True) or {}
    pairs = [(it.get('vector', []), it.get('metadata', {})) for it in data.get('items', [])]
    try:
        vids = wdbx.bulk_store(pairs)
        return jsonify({'vector_ids': vids})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/vector/bulk_search', methods=['POST'])
def api_bulk_search_vectors():
    data = request.get_json(force=True) or {}
    try:
        results = wdbx.bulk_search(data.get('vectors', []), limit=data.get('limit', 10))
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Artifact endpoints
@wdbx_api.route('/artifact/store', methods=['POST'])
def api_store_artifact():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    tmp = file.stream
    meta = {}
    try:
        meta = request.form.get('metadata') and json.loads(request.form['metadata']) or {}
    except:
        pass
    try:
        aid = wdbx.store_model(file, meta) if hasattr(wdbx, 'store_model') else None
        return jsonify({'artifact_id': aid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/artifact/load/<int:artifact_id>', methods=['GET'])
def api_load_artifact(artifact_id):
    try:
        tmp_path = wdbx.load_model(artifact_id, None)
        return send_file(tmp_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Self-update endpoints for WDBX
@wdbx_api.route('/ai_update', methods=['POST'])
def api_wdbx_ai_update():
    data = request.get_json(force=True) or {}
    try:
        wdbx.ai_update(
            data['file_path'],
            data['instruction'],
            model_name=data.get('model_name'),
            backend=data.get('backend'),
            memory_limit=data.get('memory_limit')
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/git_update', methods=['POST'])
def api_wdbx_git_update():
    data = request.get_json(force=True) or {}
    try:
        wdbx.git_update(data['local_dir'], module_paths=data.get('module_paths'))
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/schedule_self_update', methods=['POST'])
def api_wdbx_schedule_self_update():
    data = request.get_json(force=True) or {}
    try:
        wdbx.schedule_self_update(
            interval=data['interval'],
            repo_dir=data['repo_dir'],
            module_paths=data.get('module_paths')
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/stop_self_update', methods=['POST'])
def api_wdbx_stop_self_update():
    try:
        wdbx.stop_self_update()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/rollback_update', methods=['POST'])
def api_wdbx_rollback_update():
    data = request.get_json(force=True) or {}
    try:
        wdbx.rollback_update(data['file_path'], backup_file=data.get('backup_file'))
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/shards', methods=['GET'])
def api_list_shards():
    """List configured shard nodes."""
    try:
        shards = list(wdbx.shards.keys()) if hasattr(wdbx, 'shards') else []
        return jsonify({'shards': shards})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/shards/health', methods=['GET'])
def api_shards_health():
    """Get health status of each shard."""
    try:
        health = wdbx.check_shards_health() if hasattr(wdbx, 'check_shards_health') else {}
        return jsonify({'health': health})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API Key Management ---
@wdbx_api.route('/apikey/generate', methods=['POST'])
def api_generate_api_key():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    try:
        key = wdbx.generate_api_key(user_id)
        return jsonify({'api_key': key})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/apikey/revoke', methods=['POST'])
def api_revoke_api_key():
    data = request.get_json(force=True) or {}
    key = data.get('key')
    if not key:
        return jsonify({'error': 'Missing key'}), 400
    try:
        success = wdbx.revoke_api_key(key)
        return jsonify({'revoked': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Audit Logs ---
@wdbx_api.route('/audit_logs', methods=['GET'])
def api_get_audit_logs():
    since = request.args.get('since')
    try:
        logs = wdbx.get_audit_logs(since)
        return jsonify({'logs': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Advanced Features ---
@wdbx_api.route('/drift', methods=['POST'])
def api_detect_drift():
    """Detect drift in the vector store"""
    data = request.get_json(force=True) or {}
    threshold = data.get('threshold', 0.1)
    try:
        drifted = wdbx.detect_drift(threshold)
        return jsonify({'drift_detected': drifted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/personas', methods=['GET'])
def api_list_personas():
    """List configured personas"""
    try:
        personas = wdbx.list_personas()
        return jsonify({'personas': personas})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/personas', methods=['POST'])
def api_create_persona():
    """Create a new persona"""
    data = request.get_json(force=True) or {}
    name = data.get('name')
    config = data.get('config', {})
    if not name:
        return jsonify({'error': 'Missing persona name'}), 400
    try:
        pid = wdbx.create_persona(name, config)
        return jsonify({'persona_id': pid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/personas/switch', methods=['POST'])
def api_switch_persona():
    """Switch to a specified persona"""
    data = request.get_json(force=True) or {}
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Missing persona name'}), 400
    try:
        wdbx.switch_persona(name)
        return jsonify({'switched': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/filter', methods=['POST'])
def api_filter_content():
    """Filter content for safety/compliance"""
    data = request.get_json(force=True) or {}
    text = data.get('text', '')
    try:
        out = wdbx.filter_content(text)
        return jsonify({'filtered': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/mitigate', methods=['POST'])
def api_mitigate_bias():
    """Mitigate bias in text"""
    data = request.get_json(force=True) or {}
    text = data.get('text', '')
    try:
        out = wdbx.mitigate_bias(text)
        return jsonify({'mitigated': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/version', methods=['GET'])
def api_backend_version():
    """Get backend version"""
    try:
        v = wdbx.get_backend_version()
        return jsonify({'version': v})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/features/<feature>', methods=['GET'])
def api_supports_feature(feature):
    """Check feature support"""
    try:
        ok = wdbx.supports_feature(feature)
        return jsonify({'supported': ok})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/blocks', methods=['GET'])
def api_list_blocks():
    """List blockchain blocks"""
    try:
        blocks = wdbx.list_blocks()
        return jsonify({'blocks': blocks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/blocks/<int:block_id>', methods=['GET'])
def api_get_block(block_id):
    """Get block details"""
    try:
        blk = wdbx.get_block(block_id)
        return jsonify({'block': blk})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/transactions', methods=['GET'])
def api_list_transactions():
    """List MVCC transactions"""
    try:
        txs = wdbx.list_transactions()
        return jsonify({'transactions': txs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@wdbx_api.route('/transactions/<int:tx_id>', methods=['GET'])
def api_get_transaction(tx_id):
    """Get transaction details"""
    try:
        tx = wdbx.get_transaction(tx_id)
        return jsonify({'transaction': tx})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 