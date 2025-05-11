__plugin__ = "StrimEditor"

from flask_plugins import Plugin


class StrimEditor(Plugin):
    def setup(self):
        # Register the Strim Editor blueprint
        from .views import strim_bp

        self.app.register_blueprint(strim_bp)
