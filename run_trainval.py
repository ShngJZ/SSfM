from pathlib import Path
import sys
import argparse
import importlib
import random
import time
from datetime import date
from typing import Any, Dict, Optional

import cv2 as cv
import torch
import torch.backends.cudnn
from easydict import EasyDict as edict

import source.admin.settings as ws_settings
from source.training.define_trainer import define_trainer

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT))

def setup_training_paths(settings: edict, train_module: str, train_name: str, train_sub: Optional[int], scene: Optional[str]) -> str:
    """
    Set up training paths and module names.
    
    Args:
        settings: Settings object containing configuration
        train_module: Name of module in train_settings
        train_name: Name of train settings file
        train_sub: Number of input views to consider
        scene: Scene identifier
        
    Returns:
        Original train module name for launching
    """
    train_module_for_launching = train_module
    base_dir_parts = train_module.split('/')

    if train_sub is not None and train_sub != 0:
        base_dir_parts[1] += f'/subset_{train_sub}'
    
    if scene is not None:
        base_dir_parts[1] += f'/{scene}'
    
    modified_train_module = '/'.join(base_dir_parts)
    
    settings.module_name_for_eval = train_module_for_launching
    settings.module_name = modified_train_module
    settings.script_name = train_name
    settings.project_path = f'{modified_train_module}/{train_name}'
    
    return train_module_for_launching

def update_settings(settings: edict, args: argparse.Namespace) -> None:
    """
    Update settings with command line arguments and create workspace directory.
    
    Args:
        settings: Settings object to update
        args: Command line arguments
    """
    # Update settings with command line args
    args_to_update = {k: v for k, v in vars(args).items() if v is not None}
    args_to_update.update({k: v for k, v in settings.__dict__.items() 
                          if k != 'env' and v is not None})
    settings.args_to_update = args_to_update
    
    # Create workspace directory if needed
    workspace_path = Path(settings.env.workspace_dir) / settings.project_path
    workspace_path.mkdir(parents=True, exist_ok=True)

def run_training(
    train_module: str,
    train_name: str,
    seed: int,
    cudnn_benchmark: bool = True,
    data_root: str = '',
    args: Optional[Dict[str, Any]] = None
) -> None:
    """
    Run training scripts from train_settings.
    
    Args:
        train_module: Name of module in train_settings folder
        train_name: Name of train settings file
        seed: Random seed for reproducibility
        cudnn_benchmark: Whether to use cudnn benchmark
        data_root: Root directory for data (for server usage)
        debug: Enable debug mode
        args: Additional arguments for training
    """
    # Avoid opencv-related crashes
    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Initialize settings
    settings = ws_settings.Settings(data_root)
    settings.data_root = data_root
    settings.seed = seed
    settings.use_wandb = False
    settings.distributed = False  # Multi-GPU not supported
    settings.local_rank = 0

    # Set up training paths
    train_module_for_launching = setup_training_paths(
        settings, train_module, train_name, args.train_sub, args.scene
    )
    
    # Update settings with arguments
    update_settings(settings, args)
    
    # Log training information
    today = date.today()
    print(f'Training: {train_module} {train_name}\n'
          f'Date: {today.strftime("%d/%m/%Y")}')
    
    # Import and initialize training configuration
    module_path = f'train_settings.{train_module_for_launching.replace("/", ".")}'
    module_path = f'{module_path}.{train_name.replace("/", ".")}'
    expr_module = importlib.import_module(module_path)
    
    # Convert settings to EasyDict and get model config
    settings = edict(settings.__dict__)
    model_config = expr_module.get_config()
    
    # Initialize trainer
    trainer = define_trainer(args=settings, settings_model=model_config)
    
    if args.stage != -1:
        trainer.run(load_latest=True)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Run training scripts from train_settings.')
    
    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        '--train_module',
        type=str,
        default="joint_pose_nerf_training/scannet_depth_exp",
        help='Module name in train_settings folder'
    )
    training_group.add_argument(
        '--train_name',
        type=str,
        default="zoedepth_pdcnet",
        help='Name of train settings file'
    )
    training_group.add_argument(
        '--train_sub',
        type=int,
        default=5,
        help='Number of input views to consider'
    )
    
    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        '--data_root',
        type=str,
        default='/home/ubuntu/disk6/RSfM-Datasets/ScanNet',
        help='Root directory for dataset'
    )
    data_group.add_argument(
        '--scene',
        type=str,
        default="scene0708_00",
        help='Scene identifier'
    )
    data_group.add_argument(
        '--dataset_entry',
        type=str,
        default='scene0708_00 735.jpg 678.jpg 764.jpg 686.jpg 755.jpg\n',
        help='Dataset entry information'
    )
    
    # Runtime configuration
    runtime_group = parser.add_argument_group('Runtime Configuration')
    runtime_group.add_argument(
        '--cudnn_benchmark',
        type=bool,
        default=True,
        help='Enable cudnn benchmark'
    )
    runtime_group.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )
    runtime_group.add_argument(
        '--stage',
        type=int,
        default=2,
        choices=[-1, 0, 1, 2],
        help=('-1: Only init camera pose without bundle-adjustment\n'
              ' 0: Reserved\n'
              ' 1: Pose optimization\n'
              ' 2: Depth optimization')
    )
    
    return parser.parse_args()

def main() -> None:
    """Main execution function for training pipeline."""
    args = parse_args()
    
    # Random wait to avoid resource conflicts
    wait_time = random.randint(1, 120)
    time.sleep(wait_time)
    
    run_training(
        train_module=args.train_module,
        train_name=args.train_name,
        cudnn_benchmark=args.cudnn_benchmark,
        seed=args.seed,
        data_root=args.data_root,
        args=args
    )

if __name__ == '__main__':
    main()