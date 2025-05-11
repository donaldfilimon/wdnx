# Initialize plugin utilities: error handlers and health checks


def init_app(app):
    """
    Register core error handling and health check blueprints on the Flask app.
    """
    from .error_handler import error_bp
    from .health import health_bp
    from .logs import logs_bp

    app.register_blueprint(error_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(logs_bp)
    # Expose plugin code utilities


__all__ = [
    "init_app",
    "get_logger",
    "require_query_params",
    "require_json_fields",
    "require_file_param",
    "metrics",
    "get_plugin_logger",
    "plugin_metadata",
]
