__plugin__ = "MLTasks"

from flask_plugins import Plugin


class MLTasks(Plugin):
    def setup(self):
        # Register CLI commands
        from .cli import bp as ml_cli_bp

        self.app.register_blueprint(ml_cli_bp)
        # Register HTTP API routes
        from .views import ml_bp

        self.app.register_blueprint(ml_bp)
