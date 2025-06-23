"""Environment settings module for dynamic configuration loading."""

import importlib
from typing import Any

def env_settings(data_root: str = '') -> Any:
    """Load environment settings from local configuration.
    
    Args:
        data_root: Root directory for data files.
    
    Returns:
        EnvironmentSettings instance with configured paths.
    """
    env_module = importlib.import_module('source.admin.local')
    return env_module.EnvironmentSettings(data_root)
