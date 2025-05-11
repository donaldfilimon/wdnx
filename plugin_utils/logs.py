import logging
from collections import deque

from flask import Blueprint, jsonify, request

# In-memory log storage per plugin
LOG_HANDLERS: dict[str, "DequeHandler"] = {}


class DequeHandler(logging.Handler):
    """
    Logging handler that stores formatted log records in a bounded deque per plugin.
    """

    def __init__(self, plugin_id: str, capacity: int = 100):
        super().__init__()
        self.plugin_id = plugin_id
        self.deque = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.deque.append(msg)
        except Exception:
            # Avoid breaking logging in case of formatting errors
            pass


def get_plugin_logger(plugin_id: str, capacity: int = 100) -> logging.Logger:
    """
    Get or create a logger for a plugin that stores recent log messages.

    Args:
        plugin_id: Identifier of the plugin.
        capacity: Max number of log entries to keep.
    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(f"plugin.{plugin_id}")
    if plugin_id not in LOG_HANDLERS:
        handler = DequeHandler(plugin_id, capacity)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        LOG_HANDLERS[plugin_id] = handler
    return logger


# Blueprint to expose plugin logs
logs_bp = Blueprint("plugin_logs", __name__)


@logs_bp.route("/plugins/<plugin_id>/logs", methods=["GET"])
def plugin_logs(plugin_id: str):
    """
    Return recent log entries for a plugin.

    Query Parameters:
        limit (int): max number of entries to return (default 50).
    """
    limit = request.args.get("limit", default=50, type=int)
    handler = LOG_HANDLERS.get(plugin_id)
    if handler is None:
        return jsonify({"error": "Plugin logs not found"}), 404
    entries = list(handler.deque)
    # Return last 'limit' entries (most recent)
    return jsonify({"plugin": plugin_id, "logs": entries[-limit:]}), 200
