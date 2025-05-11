__plugin__ = "OllamaProvider"

from flask_plugins import Plugin


class OllamaProvider(Plugin):
    def setup(self):
        from .views import ollama_bp

        self.app.register_blueprint(ollama_bp)
