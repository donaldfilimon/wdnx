from flask import Blueprint, jsonify
from app import plugin_manager

# Health check endpoints
health_bp = Blueprint('health_check', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    """Application health endpoint"""
    return jsonify({'status': 'ok'}), 200

@health_bp.route('/plugins/<plugin_id>/health', methods=['GET'])
def plugin_health(plugin_id):
    """Check status of a specific plugin"""
    plugin = plugin_manager.all_plugins.get(plugin_id)
    if plugin and plugin.enabled:
        return jsonify({'plugin': plugin_id, 'status': 'enabled'}), 200
    return jsonify({'plugin': plugin_id, 'status': 'disabled or not found'}), 404 