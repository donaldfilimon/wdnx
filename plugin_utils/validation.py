from functools import wraps

from flask import jsonify, request

# Decorators to validate request parameters or payloads


def require_query_params(*params):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            missing = [p for p in params if request.args.get(p) is None]
            if missing:
                return (
                    jsonify({"error": "Missing query parameters", "missing": missing}),
                    400,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_json_fields(*fields):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            data = request.get_json(silent=True) or {}
            missing = [f for f in fields if f not in data]
            if missing:
                return (
                    jsonify({"error": "Missing JSON fields", "missing": missing}),
                    400,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_file_param(field_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if field_name not in request.files:
                return (
                    jsonify({"error": "Missing file parameter", "missing": field_name}),
                    400,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
