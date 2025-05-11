from flask import Blueprint, jsonify, current_app
from werkzeug.exceptions import HTTPException

# Blueprint for handling errors across all plugins
error_bp = Blueprint('error_handler', __name__)

@error_bp.app_errorhandler(HTTPException)
def handle_http_exception(e):
    """Return JSON for HTTP exceptions"""
    return jsonify({'error': e.name, 'description': e.description}), e.code

@error_bp.app_errorhandler(Exception)
def handle_generic_exception(e):
    """Return JSON for uncaught exceptions"""
    current_app.logger.exception("Unhandled exception")
    return jsonify({'error': 'Internal Server Error', 'description': str(e)}), 500 