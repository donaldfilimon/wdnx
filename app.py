"""
Main Flask application for WDBX and Lylex platform.

This script initializes the Flask app, configures authentication, plugins, metrics,
and exposes API endpoints for vector storage, artifact management, plugin control,
self-updating, and more.

Improvements:
- PEP8 formatting and import order
- Type annotations
- Comprehensive docstrings
- Enhanced error handling and logging
- Security and clarity improvements
"""

import asyncio
import importlib.metadata
import json
import logging
import os
import socket
import ssl
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import (
    Blueprint,
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
    verify_jwt_in_request,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from blueprints.lylex_api import lylex_api
from blueprints.plugin_admin import plugin_admin
from blueprints.wdbx_api import wdbx_api
from config import Settings
from lylex.db import LylexDB
from wdbx import WDBX
from wdbx.metrics import start_metrics_server

# --- Logging and Environment Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
settings = Settings()

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = settings.flask_secret_key
CORS(app)

# --- Rate Limiting ---
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute", "1000 per hour"],
)

# --- Authentication: Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    """Flask-Login user class."""
    def __init__(self, username: str):
        self.id = username

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    """Load user for Flask-Login."""
    if settings.admin_username and user_id == settings.admin_username:
        return User(user_id)
    return None

# --- JWT Authentication ---
app.config["JWT_SECRET_KEY"] = settings.jwt_secret_key
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(minutes=settings.jwt_access_expires_minutes)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=settings.jwt_refresh_expires_days)
jwt = JWTManager(app)

revoked_tokens: Set[str] = set()

@jwt.unauthorized_loader
def missing_token_callback(error: str) -> Response:
    """Handle missing JWT."""
    return jsonify({"msg": error}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error: str) -> Response:
    """Handle invalid JWT."""
    return jsonify({"msg": error}), 422

@jwt.expired_token_loader
def expired_token_callback(jwt_header: dict, jwt_payload: dict) -> Response:
    """Handle expired JWT."""
    return jsonify({"msg": "Token has expired"}), 401

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header: dict, jwt_payload: dict) -> bool:
    """Return True if this token has been revoked."""
    jti = jwt_payload.get("jti")
    return jti in revoked_tokens

def roles_required(roles: List[str]) -> Callable:
    """
    Decorator for role-based access control.
    Usage: @roles_required(['admin'])
    """
    def wrapper(fn: Callable) -> Callable:
        @wraps(fn)
        @jwt_required()
        def decorator(*args, **kwargs):
            claims = get_jwt()
            token_roles = claims.get("roles", [])
            if not any(r in token_roles for r in roles):
                return jsonify({"msg": "Forbidden"}), 403
            return fn(*args, **kwargs)
        return decorator
    return wrapper

# --- Database and Plugin Initialization ---
UPLOAD_FOLDER = settings.upload_folder
OUTPUT_FOLDER = settings.output_folder
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

wdbx = WDBX(
    vector_dimension=settings.wdbx_vector_dimension,
    enable_plugins=settings.wdbx_enable_plugins,
)
wdbx.initialize()

start_metrics_server(port=settings.metrics_port, addr=settings.metrics_addr)

plugin_manager = importlib.import_module("flask_plugins").PluginManager()
plugin_manager.init_app(app)

# Monkey-patch PluginManager to include entry-point plugins
_orig_find_plugins = plugin_manager.find_plugins
def _find_plugins_with_entrypoints(self):
    found = _orig_find_plugins(self)
    eps = importlib.metadata.entry_points().select(group="myapp.plugins")
    for ep in eps:
        try:
            pkg = ep.module
            found[ep.name] = pkg
            self._available_plugins[ep.name] = pkg
        except Exception as e:
            logger.error(f"EP plugin load fail {ep.name}: {e}")
    return found
plugin_manager.find_plugins = _find_plugins_with_entrypoints.__get__(plugin_manager)
plugin_manager.setup_plugins()

# --- LylexDB and Brain Learner ---
lylex_db = LylexDB(vector_dimension=settings.wdbx_vector_dimension, embed_fn=None)

from lylex.brain import Brain
brain = Brain(
    model_name=getattr(settings, "brain_model_name", "gpt2"),
    backend=getattr(settings, "brain_backend", "pt"),
    memory_db=lylex_db,
    memory_limit=getattr(settings, "brain_memory_limit", 100),
    interval_minutes=getattr(settings, "brain_interval_minutes", 60),
    train_epochs=getattr(settings, "brain_train_epochs", 1),
    batch_size=getattr(settings, "brain_batch_size", 1),
    mixed_precision=getattr(settings, "brain_mixed_precision", True),
    peft_r=getattr(settings, "brain_peft_r", 8),
    peft_alpha=getattr(settings, "brain_peft_alpha", 16),
    peft_dropout=getattr(settings, "brain_peft_dropout", 0.1),
    wandb_project=getattr(settings, "brain_wandb_project", None),
)

# --- API Blueprints ---
api = Blueprint("api", __name__, url_prefix="/api")

@api.before_request
def _api_require_jwt() -> Optional[Response]:
    """Enforce JWT on all API routes."""
    verify_jwt_in_request()

@api.app_errorhandler(HTTPException)
def _handle_http_exception(e: HTTPException) -> Response:
    """Return JSON for HTTP exceptions."""
    response = e.get_response()
    response.data = json.dumps({"error": e.name, "description": e.description})
    response.content_type = "application/json"
    return response

@api.app_errorhandler(Exception)
def _handle_generic_exception(e: Exception) -> Response:
    """Return JSON for uncaught exceptions."""
    logger.exception(e)
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

api.register_blueprint(wdbx_api, url_prefix="/wdbx")
api.register_blueprint(lylex_api, url_prefix="/lylex")
api.register_blueprint(plugin_admin, url_prefix="/plugins")
app.register_blueprint(api)

# --- Scheduler for Outdated Packages ---
scheduler = BackgroundScheduler()

def update_and_notify() -> int:
    """
    Record outdated packages and send notifications if configured.
    Returns the count of recorded packages.
    """
    from wdbx.update_utils import get_outdated_packages
    outdated = get_outdated_packages()
    if not outdated:
        return 0
    count = wdbx.record_outdated_packages()
    lines = [f"{pkg['name']}: {pkg['version']} -> {pkg['latest_version']}" for pkg in outdated]
    msg = "Outdated packages recorded:\n" + "\n".join(lines)
    import requests
    if settings.slack_webhook_url:
        try:
            requests.post(settings.slack_webhook_url, json={"text": msg})
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
    if settings.teams_webhook_url:
        try:
            requests.post(settings.teams_webhook_url, json={"text": msg})
        except Exception as e:
            logger.error(f"Teams notification failed: {e}")
    return count

scheduler.add_job(
    update_and_notify,
    "interval",
    minutes=settings.update_interval_minutes,
    id="outdated_job",
)
scheduler.start()
logger.info(f"Scheduled periodic outdated package recording every {settings.update_interval_minutes} minutes")
initial_count = update_and_notify()
logger.info(f"Initial outdated packages recorded into WDBX: {initial_count}")

# --- UI Blueprint for Web and API ---
ui = Blueprint("ui", __name__, template_folder="templates")

@ui.before_request
def require_login_ui() -> Optional[Response]:
    """Redirect to login if user is not authenticated."""
    if not current_user.is_authenticated:
        return redirect(url_for("login"))

@app.route("/", methods=["GET"])
def index() -> str:
    """Render the home page."""
    return render_template("index.html")

@ui.route("/ui")
def dashboard() -> str:
    """Render the dashboard page."""
    return render_template("dashboard.html")

@ui.route("/api/vector/store", methods=["POST"])
def api_store_vector() -> Response:
    """Store a vector and its metadata."""
    data = request.get_json(force=True)
    try:
        vid = wdbx.store(data.get("vector", []), data.get("metadata", {}))
        return jsonify({"vector_id": vid})
    except Exception as e:
        logger.error(f"Error storing vector: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/vector/search", methods=["POST"])
def api_search_vector() -> Response:
    """Search for similar vectors."""
    data = request.get_json(force=True)
    try:
        raw = wdbx.search(data.get("vector", []), limit=data.get("limit", 10))
        results = [{"id": r[0], "score": r[1], "metadata": r[2]} for r in raw]
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error searching vector: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/artifact/store", methods=["POST"])
def api_store_artifact() -> Response:
    """Store a model artifact file."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    fname = secure_filename(f.filename)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        f.save(tmp.name)
        meta = {}
        try:
            meta = json.loads(request.form.get("metadata", "{}"))
        except Exception:
            pass
        try:
            aid = wdbx.store_model(tmp.name, meta)
            return jsonify({"artifact_id": aid})
        except Exception as e:
            logger.error(f"Error storing artifact: {e}")
            return jsonify({"error": str(e)}), 500

@ui.route("/api/artifact/load/<int:artifact_id>")
def api_load_artifact(artifact_id: int) -> Response:
    """Download a stored model artifact."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        wdbx.load_model(artifact_id, tmp.name)
        return send_file(tmp.name, as_attachment=True, download_name=f"artifact_{artifact_id}.bin")

@ui.route("/api/lylex/store_interaction", methods=["POST"])
def api_store_interaction() -> Response:
    """Store a prompt-response interaction."""
    try:
        data = request.get_json(force=True)
        lid = lylex_db.store_interaction(
            data.get("prompt", ""),
            data.get("response", ""),
            data.get("metadata", {}),
        )
        return jsonify({"interaction_id": lid})
    except Exception as e:
        logger.error(f"Error storing interaction: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/search_interactions", methods=["POST"])
def api_search_interactions() -> Response:
    """Search for similar interactions."""
    try:
        data = request.get_json(force=True)
        raw = lylex_db.search_interactions(data.get("prompt", ""), limit=data.get("limit", 5))
        results = [{"id": r[0], "score": r[1], "metadata": r[2]} for r in raw]
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error searching interactions: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/neural_backtrace", methods=["POST"])
def api_neural_backtrace() -> Response:
    """Perform a neural backtrace for a prompt."""
    try:
        data = request.get_json(force=True)
        info = lylex_db.neural_backtrace(data.get("prompt", ""))
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error in neural backtrace: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/vector/bulk_store", methods=["POST"])
def api_bulk_store_vectors() -> Response:
    """Bulk store vectors."""
    data = request.get_json(force=True) or {}
    items = data.get("items", [])
    pairs = [(it.get("vector", []), it.get("metadata", {})) for it in items]
    vids = wdbx.bulk_store(pairs)
    return jsonify({"vector_ids": vids})

@ui.route("/api/vector/bulk_search", methods=["POST"])
def api_bulk_search_vectors() -> Response:
    """Bulk search vectors."""
    data = request.get_json(force=True) or {}
    vectors = data.get("vectors", [])
    limit = data.get("limit", 10)
    results = wdbx.bulk_search(vectors, limit=limit)
    return jsonify({"results": results})

# --- Self-update and Git endpoints ---
@ui.route("/api/wdbx/ai_update", methods=["POST"])
def api_wdbx_ai_update() -> Response:
    """AI-driven code update for WDBX."""
    data = request.get_json(force=True)
    try:
        wdbx.ai_update(
            data["file_path"],
            data["instruction"],
            model_name=data.get("model_name", "gpt2"),
            backend=data.get("backend", "pt"),
            memory_limit=data.get("memory_limit", 5),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error in WDBX AI update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/ai_update", methods=["POST"])
def api_lylex_ai_update() -> Response:
    """AI-driven code update for Lylex."""
    data = request.get_json(force=True)
    try:
        lylex_db.ai_update(
            data["file_path"],
            data["instruction"],
            model_name=data.get("model_name", "gpt2"),
            backend=data.get("backend", "pt"),
            memory_limit=data.get("memory_limit", 5),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error in Lylex AI update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/wdbx/git_update", methods=["POST"])
def api_wdbx_git_update() -> Response:
    """Git update for WDBX."""
    data = request.get_json(force=True)
    try:
        wdbx.git_update(
            data["local_dir"],
            module_paths=data.get("module_paths"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error in WDBX git update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/git_update", methods=["POST"])
def api_lylex_git_update() -> Response:
    """Git update for Lylex."""
    data = request.get_json(force=True)
    try:
        lylex_db.git_update(
            data["local_dir"],
            module_paths=data.get("module_paths"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error in Lylex git update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/wdbx/schedule_self_update", methods=["POST"])
def api_wdbx_schedule_self_update() -> Response:
    """Schedule self-update for WDBX."""
    data = request.get_json(force=True)
    try:
        wdbx.schedule_self_update(
            interval=data["interval"],
            repo_dir=data["repo_dir"],
            module_paths=data.get("module_paths"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error scheduling WDBX self-update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/schedule_self_update", methods=["POST"])
def api_lylex_schedule_self_update() -> Response:
    """Schedule self-update for Lylex."""
    data = request.get_json(force=True)
    try:
        lylex_db.schedule_self_update(
            interval=data["interval"],
            repo_dir=data["repo_dir"],
            module_paths=data.get("module_paths"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error scheduling Lylex self-update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/wdbx/stop_self_update", methods=["POST"])
def api_wdbx_stop_self_update() -> Response:
    """Stop WDBX self-update."""
    try:
        wdbx.stop_self_update()
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error stopping WDBX self-update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/stop_self_update", methods=["POST"])
def api_lylex_stop_self_update() -> Response:
    """Stop Lylex self-update."""
    try:
        lylex_db.stop_self_update()
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error stopping Lylex self-update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/wdbx/rollback_update", methods=["POST"])
def api_wdbx_rollback_update() -> Response:
    """Rollback WDBX update."""
    data = request.get_json(force=True)
    try:
        wdbx.rollback_update(
            data["file_path"],
            backup_file=data.get("backup_file"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error rolling back WDBX update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/lylex/rollback_update", methods=["POST"])
def api_lylex_rollback_update() -> Response:
    """Rollback Lylex update."""
    data = request.get_json(force=True)
    try:
        lylex_db.rollback_update(
            data["file_path"],
            backup_file=data.get("backup_file"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Error rolling back Lylex update: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/updates")
def api_updates() -> Response:
    """Return stored outdated package metadata."""
    try:
        data = lylex_db.search_outdated_packages()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching updates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/updates")
def view_updates() -> str:
    """Render a simple HTML page listing outdated packages."""
    try:
        packages = lylex_db.search_outdated_packages()
    except Exception as e:
        flash(f"Failed to load updates: {e}")
        return redirect(url_for("index"))
    return render_template("updates.html", packages=packages)

# --- Scheduler Control Endpoints ---
@ui.route("/api/scheduler/pause", methods=["POST"])
def api_scheduler_pause() -> Response:
    """Pause the outdated package scheduler."""
    try:
        scheduler.pause_job("outdated_job")
        return jsonify({"status": "paused"})
    except Exception as e:
        logger.error(f"Error pausing scheduler: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/scheduler/resume", methods=["POST"])
def api_scheduler_resume() -> Response:
    """Resume the outdated package scheduler."""
    try:
        scheduler.resume_job("outdated_job")
        return jsonify({"status": "resumed"})
    except Exception as e:
        logger.error(f"Error resuming scheduler: {e}")
        return jsonify({"error": str(e)}), 500

@ui.route("/api/scheduler/interval", methods=["POST"])
def api_scheduler_set_interval() -> Response:
    """Set the interval for the outdated package scheduler."""
    data = request.get_json(force=True)
    interval = data.get("interval")
    try:
        scheduler.reschedule_job("outdated_job", trigger="interval", minutes=interval)
        return jsonify({"status": "interval updated", "interval": interval})
    except Exception as e:
        logger.error(f"Error setting scheduler interval: {e}")
        return jsonify({"error": str(e)}), 500

app.register_blueprint(ui)
app.register_blueprint(plugin_admin)

# --- Authentication Routes ---
@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    """Login page and handler."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if (
            username
            and password
            and username == settings.admin_username
            and settings.admin_password_hash
            and check_password_hash(settings.admin_password_hash, password)
        ):
            user = User(username)
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout() -> Response:
    """Logout the current user."""
    logout_user()
    return redirect(url_for("login"))

# --- JWT Authentication Endpoints ---
@limiter.limit("5 per minute")
@app.route("/auth/login", methods=["POST"])
def auth_login() -> Response:
    """JWT login endpoint."""
    data = request.get_json(force=True) or {}
    username = data.get("username")
    password = data.get("password")
    if (
        username == settings.admin_username
        and settings.admin_password_hash
        and check_password_hash(settings.admin_password_hash, password)
    ):
        roles = settings.admin_roles if username == settings.admin_username else []
        access = create_access_token(identity=username, additional_claims={"roles": roles})
        refresh = create_refresh_token(identity=username, additional_claims={"roles": roles})
        return jsonify(access_token=access, refresh_token=refresh)
    return jsonify({"msg": "Bad username or password"}), 401

@app.route("/auth/refresh", methods=["POST"])
@jwt_required(refresh=True)
def auth_refresh() -> Response:
    """Refresh JWT access token."""
    identity = get_jwt_identity()
    claims = get_jwt()
    roles = claims.get("roles", [])
    new_access = create_access_token(identity=identity, additional_claims={"roles": roles})
    return jsonify(access_token=new_access)

@app.route("/auth/logout/access", methods=["DELETE"])
@jwt_required()
def logout_access() -> Response:
    """Revoke the current access token."""
    jti = get_jwt()["jti"]
    revoked_tokens.add(jti)
    return jsonify({"msg": "Access token revoked"}), 200

@app.route("/auth/logout/refresh", methods=["DELETE"])
@jwt_required(refresh=True)
def logout_refresh() -> Response:
    """Revoke the current refresh token."""
    jti = get_jwt()["jti"]
    revoked_tokens.add(jti)
    return jsonify({"msg": "Refresh token revoked"}), 200

# --- SSL Socket Server Helper ---
def handle_client(conn: socket.socket) -> None:
    """Handle a single SSL client connection."""
    try:
        data = conn.recv(1024)
        conn.sendall(b"Echo: " + data)
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        conn.close()

def run_ssl_socket_server() -> None:
    """Start a simple SSL-wrapped TCP echo server if configured."""
    if settings.socket_port and settings.ssl_certfile and settings.ssl_keyfile:
        bindsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bindsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bindsocket.bind((settings.host, settings.socket_port))
        bindsocket.listen(5)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(settings.ssl_certfile), keyfile=str(settings.ssl_keyfile))
        logger.info(f"SSL socket server listening on {settings.host}:{settings.socket_port}")
        while True:
            newsocket, addr = bindsocket.accept()
            try:
                conn = context.wrap_socket(newsocket, server_side=True)
            except Exception as e:
                logger.error(f"SSL wrap failed: {e}")
                newsocket.close()
                continue
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

# --- Main Entrypoint ---
if __name__ == "__main__":
    try:
        threading.Thread(target=run_ssl_socket_server, daemon=True).start()
        ssl_context = None
        if settings.ssl_certfile and settings.ssl_keyfile:
            ssl_context = (str(settings.ssl_certfile), str(settings.ssl_keyfile))
        app.run(host=settings.host, port=settings.port, ssl_context=ssl_context, debug=True)
    finally:
        wdbx.shutdown()

# --- Anchors and Metrics Endpoints ---
@app.route("/anchors")
def anchors() -> str:
    """Render a page listing anchored file hashes stored in WDBX."""
    try:
        entries = wdbx.list_anchors(limit=100)
    except Exception as e:
        flash(f"Failed to retrieve anchors: {e}")
        entries = []
    return render_template("anchors.html", anchors=entries)

@app.route("/metrics")
def metrics_endpoint() -> Response:
    """Expose Prometheus metrics."""
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

# --- Brain Control Endpoints ---
@api.route("/brain/status")
def api_brain_status() -> Response:
    """Return Brain learner scheduler status."""
    job = brain.scheduler.get_job(f"brain_{brain.model_name}")
    return jsonify({"scheduled": bool(job)})

@api.route("/brain/stop", methods=["POST"])
def api_brain_stop() -> Response:
    """Stop the Brain autonomous learning."""
    brain.stop()
    return jsonify({"status": "stopped"})

@api.route("/brain/start", methods=["POST"])
def api_brain_start() -> Response:
    """Start the Brain autonomous learning."""
    try:
        brain.scheduler.start()
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Error starting Brain: {e}")
        return jsonify({"error": str(e)}), 500
