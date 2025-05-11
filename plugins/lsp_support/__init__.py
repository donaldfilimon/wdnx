__plugin__ = "LSPSupport"

from flask_plugins import Plugin


class LSPSupport(Plugin):
    def setup(self):
        from .views import lsp_bp

        self.app.register_blueprint(lsp_bp)
