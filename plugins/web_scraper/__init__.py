__plugin__ = "WebScraper"

from flask_plugins import Plugin


class WebScraper(Plugin):
    def setup(self):
        # Register the web scraper blueprint
        from .views import scraper_bp

        self.app.register_blueprint(scraper_bp)
