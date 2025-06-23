"""Local environment configuration for paths and directories."""

from dataclasses import dataclass
from pathlib import Path

PRJ_ROOT = Path(__file__).parent.parent.parent.resolve()

@dataclass
class EnvironmentSettings:
    """Environment settings for project paths and directories."""
    
    data_root: str = ''
    
    def __post_init__(self):
        """Initialize project directories after instance creation."""
        checkpoint_dir = PRJ_ROOT / 'checkpoint'
        
        self.pretrained_networks = checkpoint_dir
        self.workspace_dir = checkpoint_dir
        self.tensorboard_dir = checkpoint_dir
        self.eval_dir = checkpoint_dir
        self.log_dir = checkpoint_dir
        self.scannet = Path(self.data_root) / 'scans_test'
