"""
Plugin Admin Blueprint

Defines the Flask blueprint for plugin administration, providing endpoints to list, enable, disable, and reload plugins with JWT-based admin access control.
"""
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt
from app import plugin_manager, app

plugin_admin = Blueprint('plugin_admin', __name__, url_prefix='/plugins')


def reload_all_plugins():
    # Clear cached plugin lists and reload plugins
    plugin_manager._plugins = None
    plugin_manager._all_plugins = None
    plugin_manager._available_plugins = {}
    plugin_manager._found_plugins = {}
    plugin_manager.setup_plugins()

@plugin_admin.before_request
@jwt_required()
def require_admin():
    """Ensure only users with 'admin' role can manage plugins."""
    claims = get_jwt()
    roles = claims.get('roles', [])
    if 'admin' not in roles:
        return jsonify({'msg': 'Admins only'}), 403

@plugin_admin.route('/status', methods=['GET'])
@jwt_required()
def plugins_status():
    """Return status of all plugins"""
    status = []
    for plugin in plugin_manager.all_plugins.values():
        status.append({
            'identifier': plugin.identifier,
            'name': plugin.name,
            'enabled': plugin.enabled,
            'version': plugin.version,
            'description': plugin.description
        })
    return jsonify({'plugins': status})

@plugin_admin.route('/enable', methods=['POST'])
@jwt_required()
def plugins_enable():
    data = request.get_json(force=True) or {}
    identifier = data.get('plugin')
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({'error': 'Plugin not found'}), 404
    try:
        plugin_manager.enable_plugins([plugin])
        plugin.setup()
        reload_all_plugins()
        return jsonify({'enabled': True, 'identifier': identifier})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@plugin_admin.route('/disable', methods=['POST'])
@jwt_required()
def plugins_disable():
    data = request.get_json(force=True) or {}
    identifier = data.get('plugin')
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({'error': 'Plugin not found'}), 404
    try:
        plugin_manager.disable_plugins([plugin])
        reload_all_plugins()
        return jsonify({'disabled': True, 'identifier': identifier})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@plugin_admin.route('/reload', methods=['POST'])
@jwt_required()
def plugins_reload():
    data = request.get_json(force=True) or {}
    identifier = data.get('plugin')
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({'error': 'Plugin not found'}), 404
    try:
        plugin_manager.disable_plugins([plugin])
        plugin_manager.enable_plugins([plugin])
        plugin.setup()
        reload_all_plugins()
        return jsonify({'reloaded': True, 'identifier': identifier})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 