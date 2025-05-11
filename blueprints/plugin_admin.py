"""
Plugin Admin Blueprint

Defines the Flask blueprint for plugin administration, providing endpoints to list, enable, disable, and reload plugins with JWT-based admin access control.
"""

from flask import Blueprint, Response, jsonify, request
from flask_jwt_extended import get_jwt, jwt_required

from app import plugin_manager

plugin_admin = Blueprint("plugin_admin", __name__, url_prefix="/plugins")


def reload_all_plugins() -> None:
    # Clear cached plugin lists and reload plugins
    plugin_manager._plugins = None
    plugin_manager._all_plugins = None
    plugin_manager._available_plugins = {}
    plugin_manager._found_plugins = {}
    plugin_manager.setup_plugins()


@plugin_admin.before_request
@jwt_required()
def require_admin() -> Response:
    """
    Ensure only users with 'admin' role can manage plugins.

    Returns:
        Response: JSON {"msg": "Admins only"} with HTTP 403 if unauthorized.
    """
    claims = get_jwt()
    roles = claims.get("roles", [])
    if "admin" not in roles:
        return jsonify({"msg": "Admins only"}), 403


@plugin_admin.route("/status", methods=["GET"])
@jwt_required()
def plugins_status() -> Response:
    """
    Return status of all registered plugins.

    Returns:
        Response: JSON {"plugins": list[dict]} where each dict contains plugin metadata.

    Raises:
        Exception: If retrieval fails (HTTP 500).
    """
    status = []
    for plugin in plugin_manager.all_plugins.values():
        status.append(
            {
                "identifier": plugin.identifier,
                "name": plugin.name,
                "enabled": plugin.enabled,
                "version": plugin.version,
                "description": plugin.description,
            }
        )
    return jsonify({"plugins": status})


@plugin_admin.route("/enable", methods=["POST"])
@jwt_required()
def plugins_enable() -> Response:
    """
    Enable a specified plugin by identifier.

    Args:
        plugin (str): Identifier of the plugin to enable (from JSON body under "plugin").

    Returns:
        Response: JSON {"enabled": bool, "identifier": str}.

    Raises:
        404: If plugin not found.
        Exception: If enabling fails (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    identifier = data.get("plugin")
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({"error": "Plugin not found"}), 404
    try:
        plugin_manager.enable_plugins([plugin])
        plugin.setup()
        reload_all_plugins()
        return jsonify({"enabled": True, "identifier": identifier})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@plugin_admin.route("/disable", methods=["POST"])
@jwt_required()
def plugins_disable() -> Response:
    """
    Disable a specified plugin by identifier.

    Args:
        plugin (str): Identifier of the plugin to disable (from JSON body under "plugin").

    Returns:
        Response: JSON {"disabled": bool, "identifier": str}.

    Raises:
        404: If plugin not found.
        Exception: If disabling fails (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    identifier = data.get("plugin")
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({"error": "Plugin not found"}), 404
    try:
        plugin_manager.disable_plugins([plugin])
        reload_all_plugins()
        return jsonify({"disabled": True, "identifier": identifier})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@plugin_admin.route("/reload", methods=["POST"])
@jwt_required()
def plugins_reload() -> Response:
    """
    Reload a specified plugin by identifier.

    Args:
        plugin (str): Identifier of the plugin to reload (from JSON body under "plugin").

    Returns:
        Response: JSON {"reloaded": bool, "identifier": str}.

    Raises:
        404: If plugin not found.
        Exception: If reload fails (HTTP 500).
    """
    data = request.get_json(force=True) or {}
    identifier = data.get("plugin")
    plugin = plugin_manager.all_plugins.get(identifier)
    if not plugin:
        return jsonify({"error": "Plugin not found"}), 404
    try:
        plugin_manager.disable_plugins([plugin])
        plugin_manager.enable_plugins([plugin])
        plugin.setup()
        reload_all_plugins()
        return jsonify({"reloaded": True, "identifier": identifier})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
