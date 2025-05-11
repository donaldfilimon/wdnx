from pathlib import Path
from typing import List, Optional

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    flask_secret_key: str = "replace-me-with-secure-key"
    upload_folder: Path = Path("uploads")
    output_folder: Path = Path("generated_html")
    wdbx_vector_dimension: int = 1
    wdbx_enable_plugins: bool = True
    metrics_port: int = 8000
    metrics_addr: str = "0.0.0.0"
    update_interval_minutes: int = 60
    wdbx_repo_url: Optional[str] = None
    lylex_repo_url: Optional[str] = None
    # Admin credentials for secure access via Flask-Login
    admin_username: Optional[str] = None
    admin_password_hash: Optional[str] = None
    # Roles for admin user (JWT 'roles' claim)
    admin_roles: List[str] = ["admin"]
    # Secret key for JWT token generation
    jwt_secret_key: str = "replace-me-with-jwt-secret-key"
    # JWT expiration settings (access token minutes, refresh token days)
    jwt_access_expires_minutes: int = 15
    jwt_refresh_expires_days: int = 30
    slack_webhook_url: Optional[str] = None
    teams_webhook_url: Optional[str] = None
    # HTTP server settings
    host: str = "0.0.0.0"
    port: int = 5000
    ssl_certfile: Optional[Path] = None
    ssl_keyfile: Optional[Path] = None
    # Raw SSL socket server port (optional)
    socket_port: Optional[int] = None
    # API URLs for local LLM providers
    ollama_api_url: Optional[str] = "http://localhost:11434"
    lmstudio_api_url: Optional[str] = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["Settings"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Finalized: environment variables take highest priority, then init, dotenv, file secrets
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def get_package_version() -> str:
    """
    Returns the version string from setup.py for use in the application.
    """
    import os
    import re

    setup_path = os.path.join(os.path.dirname(__file__), "setup.py")
    version_pattern = re.compile(r'version\s*=\s*[\'"]([^\'"]+)[\'"]')
    try:
        with open(setup_path, "r", encoding="utf-8") as f:
            for line in f:
                match = version_pattern.search(line)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return "0.0.0"
