__plugin__ = "DiscordBot"

import threading

from flask_plugins import Plugin

from .bot import run_discord_bot


class DiscordBot(Plugin):
    def setup(self):
        """
        Start the Discord bot as a background thread when the plugin is initialized.
        """
        thread = threading.Thread(target=run_discord_bot, daemon=True)
        thread.start()
