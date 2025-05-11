__plugin__ = "ReactiveWeb"

from flask_plugins import Plugin

class ReactiveWeb(Plugin):
    def setup(self):
        # Register the Reactive Web blueprint for dynamic content
        from .views import reactive_bp
        self.app.register_blueprint(reactive_bp) 