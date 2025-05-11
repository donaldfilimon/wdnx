import json
import subprocess
import sys
from typing import Any, Dict, List


def get_outdated_packages() -> List[Dict[str, Any]]:
    """
    Return a list of outdated pip packages as dictionaries with keys name, version, latest_version, latest_filetype.
    """
    try:
        result = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(result)
    except Exception:
        return []
