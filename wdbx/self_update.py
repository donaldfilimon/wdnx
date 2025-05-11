"""
self_update.py - Runtime code patching and module reload utilities for WDBX.
"""
import os
import logging
import importlib
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class CodeUpdater:
    """
    Utility to apply code patches to files and reload Python modules at runtime.
    """

    def apply_patch(self, file_path: str, new_code: str) -> None:
        """
        Overwrite the specified file with new_code.
        """
        abs_path = os.path.abspath(file_path)
        logger.info(f"Applying code patch to {abs_path}")
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
        logger.info(f"Patch applied to {abs_path}")

    def reload_module(self, module_name: str) -> None:
        """
        Reload a module by name after its source has been updated.
        """
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
            logger.info(f"Module '{module_name}' reloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to reload module '{module_name}': {e}")

    def self_update(self, patches: Dict[str, str]) -> None:
        """
        Apply multiple patches (file_path -> new_code) and reload associated modules.
        Expects keys like 'module_name:file_path'.
        """
        for key, code in patches.items():
            if ':' in key:
                module_name, file_path = key.split(':', 1)
            else:
                module_name, file_path = None, key
            self.apply_patch(file_path, code)
            if module_name:
                self.reload_module(module_name) 