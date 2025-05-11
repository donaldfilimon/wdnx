"""
Plugin registration for the WDBX CLI.
"""
from flask_plugins import Plugin

class WDBXCLI(Plugin):
    def setup(self):
        from .cli import wdbx_cli # Import the Click group
        self.app.cli.add_command(wdbx_cli) # Add it to Flask's CLI runner

__plugin__ = WDBXCLI # Ensure the class name matches for discovery 