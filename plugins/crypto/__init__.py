__plugin__ = "Crypto"

from flask_plugins import Plugin


class Crypto(Plugin):
    def setup(self):
        from .views import crypto_bp

        self.app.register_blueprint(crypto_bp)
