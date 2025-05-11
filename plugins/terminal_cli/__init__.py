__plugin__ = "TerminalCLI"

from flask_plugins import Plugin


class TerminalCLI(Plugin):
    def setup(self):
        # Register the terminal CLI commands blueprint
        from .cli import bp

        self.app.register_blueprint(bp)
