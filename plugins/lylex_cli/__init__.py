__plugin__ = "LylexCLI"

from flask_plugins import Plugin

class LylexCLI(Plugin):
    def setup(self):
        from .cli import bp
        self.app.register_blueprint(bp) 