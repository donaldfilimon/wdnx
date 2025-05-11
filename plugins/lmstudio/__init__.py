__plugin__ = "LMStudioProvider"

from flask_plugins import Plugin

class LMStudioProvider(Plugin):
    def setup(self):
        from .views import lmstudio_bp
        self.app.register_blueprint(lmstudio_bp) 