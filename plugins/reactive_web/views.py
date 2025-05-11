from flask import Blueprint, jsonify, render_template_string, request

reactive_bp = Blueprint("reactive_web", __name__, url_prefix="/reactive")


@reactive_bp.route("/render", methods=["POST"])
def render_dynamic():
    """
    Render a dynamic template via POST JSON:
    ```
    { "template": "<h1>Hello {{ name }}</h1>", "context": { "name": "World" } }
    ```
    """
    data = request.get_json(force=True) or {}
    template = data.get("template")
    context = data.get("context", {})
    if template is None:
        return jsonify({"error": "No template provided."}), 400
    try:
        html = render_template_string(template, **context)
        return html
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reactive_bp.route("/greet/<string:name>")
def greet(name):
    """
    Simple greeting endpoint: /reactive/greet/YourName
    """
    html = render_template_string("<h1>Hello, {{ name }}!</h1>", name=name)
    return html
