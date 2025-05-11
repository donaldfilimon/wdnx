__plugin__ = "DiscordBot"

from flask_plugins import Plugin
import threading
from .bot import run_discord_bot


class DiscordBot(Plugin):
    def setup(self):
        """
        Start the Discord bot as a background thread when the plugin is initialized.
        """
        thread = threading.Thread(target=run_discord_bot, daemon=True)
        thread.start() 