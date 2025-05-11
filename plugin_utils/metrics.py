import time
from functools import wraps

from flask import request
from prometheus_client import Counter, Histogram

# Metrics definitions
REQUEST_COUNTER = Counter(
    "plugin_request_total",
    "Total plugin HTTP requests",
    ["plugin", "endpoint", "method"],
)
REQUEST_LATENCY = Histogram(
    "plugin_request_latency_seconds",
    "Plugin request latency",
    ["plugin", "endpoint", "method"],
)
ERROR_COUNTER = Counter(
    "plugin_request_errors_total",
    "Total plugin HTTP errors",
    ["plugin", "endpoint", "method"],
)

# Decorator to instrument routes


def metrics(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        plugin = request.blueprint or "unknown"
        endpoint = request.endpoint or fn.__name__
        method = request.method
        labels = {"plugin": plugin, "endpoint": endpoint, "method": method}
        REQUEST_COUNTER.labels(**labels).inc()
        start = time.time()
        try:
            return fn(*args, **kwargs)
        except Exception:
            ERROR_COUNTER.labels(**labels).inc()
            raise
        finally:
            REQUEST_LATENCY.labels(**labels).observe(time.time() - start)

    return wrapper
