__plugin__ = "PDFConverter"

from flask_plugins import Plugin


class PDFConverter(Plugin):
    def setup(self):
        # Register the PDF converter blueprint
        from .views import pdf_bp

        self.app.register_blueprint(pdf_bp)
