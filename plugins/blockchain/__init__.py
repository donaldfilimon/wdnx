__plugin__ = "Blockchain"

from flask_plugins import Plugin


class Blockchain(Plugin):
    def setup(self):
        from .views import blockchain_bp

        self.app.register_blueprint(blockchain_bp)
