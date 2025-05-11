"""
self_update_advanced.py - Advanced runtime self-update manager with backups, diffs, rollback, Git integration, and scheduling.
"""

import importlib
import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Directory to store backups
backup_dir = os.path.join(os.getcwd(), ".code_backups")
os.makedirs(backup_dir, exist_ok=True)


class AdvancedUpdater:
    """
    Advanced self-update manager with:
    - File backups
    - Unified diff-based patching
    - Syntax validation
    - Rollback support
    - Git repository update/apply
    - Scheduled periodic checks
    """

    def __init__(self, repo_url: Optional[str] = None, branch: str = "main"):
        self.repo_url = repo_url
        self.branch = branch
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _backup(self, file_path: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = os.path.basename(file_path)
        dest = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
        shutil.copy2(file_path, dest)
        logger.info(f"Backup created: {dest}")
        return dest

    def apply_patch(self, file_path: str, new_code: str) -> None:
        """
        Apply a unified-diff patch: back up original, write new code, validate syntax.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Cannot patch non-existent file: {file_path}")
        original = open(file_path, "r", encoding="utf-8").read()
        if original == new_code:
            logger.info(f"No changes for {file_path}")
            return
        # backup
        self._backup(file_path)
        # write new
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        # syntax check
        try:
            compile(new_code, file_path, "exec")
        except Exception as e:
            logger.error(f"Syntax error after patch: {e}")
            self.rollback(file_path)
            raise
        logger.info(f"Patched file {file_path}")

    def rollback(self, file_path: str, backup_file: Optional[str] = None) -> None:
        """
        Roll back to last backup.
        """
        if not backup_file:
            # find latest backup
            files = sorted(
                f
                for f in os.listdir(backup_dir)
                if f.startswith(os.path.basename(file_path))
            )
            if not files:
                logger.error("No backups to rollback")
                return
            backup_file = os.path.join(backup_dir, files[-1])
        shutil.copy2(backup_file, file_path)
        logger.info(f"Rollback from {backup_file} applied to {file_path}")

    def reload_module(self, module_name: str) -> None:
        """
        Reload a Python module by name.
        """
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            logger.info(f"Module {module_name} reloaded.")
        except Exception as e:
            logger.error(f"Failed to reload {module_name}: {e}")

    def update_from_git(
        self, local_dir: str, module_paths: Optional[List[str]] = None
    ) -> None:
        """
        Clone/pull a Git repo and apply changes for specified module files.
        """
        if not self.repo_url:
            raise ValueError("repo_url not configured")
        if not os.path.isdir(local_dir):
            subprocess.check_call(
                ["git", "clone", "-b", self.branch, self.repo_url, local_dir]
            )
        else:
            subprocess.check_call(["git", "-C", local_dir, "fetch"])
            subprocess.check_call(["git", "-C", local_dir, "checkout", self.branch])
            subprocess.check_call(["git", "-C", local_dir, "pull"])
        logger.info(f"Repo updated at {local_dir}")
        if module_paths:
            for path in module_paths:
                rp = os.path.join(local_dir, path)
                code = open(rp, "r", encoding="utf-8").read()
                self.apply_patch(path, code)
                module_name = path.replace("/", ".").rsplit(".py", 1)[0]
                self.reload_module(module_name)

    def schedule(self, interval: int, func, *args, **kwargs) -> None:
        """
        Schedule a function to run periodically in background thread.
        """

        def runner():
            while not self._stop_event.wait(interval):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Scheduled update error: {e}")

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        logger.info("Scheduled updater started.")

    def stop(self) -> None:
        """
        Stop the scheduled background thread.
        """
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            logger.info("Scheduled updater stopped.")

    def ai_update(
        self,
        file_path: str,
        instruction: str,
        model_name: str = "gpt2",
        backend: str = "pt",
        memory_limit: int = 5,
    ) -> None:
        """
        Use an AI agent to generate and apply a patch based on the given instruction.
        """
        # Read current code
        with open(file_path, "r", encoding="utf-8") as f:
            original = f.read()
        # Build prompt for AI
        prompt = (
            f"### File: {file_path}\n"
            f"```python\n{original}\n```\n"
            f"### Instruction: {instruction}\n"
            "Provide the full updated source file without additional commentary."
        )
        # Invoke LylexAgent to generate updated code
        from lylex.ai import LylexAgent

        agent = LylexAgent(
            model_name=model_name,
            backend=backend,
            memory_db=None,
            memory_limit=memory_limit,
        )
        new_code = agent.chat(prompt)
        # Apply and validate patch
        self.apply_patch(file_path, new_code)
        # Reload module
        module_name = file_path.replace(os.sep, ".").rsplit(".py", 1)[0]
        self.reload_module(module_name)
