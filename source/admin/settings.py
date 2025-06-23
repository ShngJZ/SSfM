"""Global settings module for training configuration."""

from dataclasses import dataclass
from typing import Any, Dict

from source.admin.environment import env_settings

@dataclass
class Settings:
    """Training settings including paths to datasets and networks.
    
    Attributes:
        data_root: Root directory for data files.
        env: Environment settings instance.
        use_gpu: Whether to use GPU for training.
    """
    
    data_root: str = ''
    env: Any = None
    use_gpu: bool = True
    
    def __post_init__(self):
        """Initialize environment settings after instance creation."""
        if self.env is None:
            self.env = env_settings(self.data_root)
    
    def update(self, settings: Dict[str, Any]) -> None:
        """Update settings from a dictionary.
        
        Args:
            settings: Dictionary of setting names and values to update.
        """
        for name, value in settings.items():
            setattr(self, name, value)
