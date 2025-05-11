"""
download.py - Utilities for remote PDF downloading and optional database configuration.
"""

import logging
from pathlib import Path, PurePath
import requests

logger = logging.getLogger(__name__)

def configure_database(db_url: str) -> None:
    """
    Stub for database configuration. Placeholder for actual implementation.

    Parameters:
        db_url: Database connection URL.
    """
    logger.info(f"configure_database called with URL: {db_url} (no-op)")


def download_file(url: str, dest_dir: PurePath) -> str:
    """
    Download a file from a URL to the destination directory.

    Parameters:
        url: Remote file URL.
        dest_dir: Directory to save the file.

    Returns:
        The local file path of the downloaded file.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        requests.RequestException: For network-related errors.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    local_filename = url.split("/")[-1] or "downloaded_file"
    file_path = dest_path / local_filename
    logger.info(f"Downloading '%s' to '%s'", url, file_path)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Downloaded file to '%s'", file_path)
    return str(file_path) 