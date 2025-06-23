from pathlib import Path
import sys
import subprocess
import random
import time
import os
import argparse
from typing import Tuple, Optional

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

def get_split_path() -> Path:
    """Get the path to the dataset split file."""
    return PROJ_ROOT / 'split' / 'scannet' / 'scannet.txt'

def select_next_sequence(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    """
    Select the next sequence to process from the dataset.
    
    Args:
        args: Command line arguments containing training configuration
        
    Returns:
        Tuple of (sequence_id, dataset_entry) or (None, None) if no sequences available
    """
    split_path = get_split_path()
    with open(split_path) as file:
        entries = file.readlines()
    
    random.seed(os.getpid())
    random.shuffle(entries)

    for entry in entries:
        seq, *_ = entry.rstrip('\n').split(' ')
        output_root = PROJ_ROOT / 'checkpoint' / args.train_module / f'subset_{args.train_sub}' / seq / args.train_name
        
        if not output_root.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            return seq, entry

    # If no empty directories found, wait and check for empty content
    time.sleep(60)
    for entry in entries:
        seq, *_ = entry.rstrip('\n').split(' ')
        output_root = PROJ_ROOT / 'checkpoint' / args.train_module / f'subset_{args.train_sub}' / seq / args.train_name
        
        if output_root.exists() and not list(output_root.iterdir()):
            print(f"{seq} has empty folder after waiting one minute - regenerating...")
            return seq, entry

    return None, None

def run_optimization_stage(args: argparse.Namespace, stage: int) -> None:
    """
    Run a specific optimization stage using run_trainval.py.
    
    Args:
        args: Command line arguments
        stage: Stage number (1 for pose optimization, 2 for depth optimization)
    """
    stage_name = "Pose" if stage == 1 else "Depth"
    print(f"============= Stage {stage} {stage_name} Optimization =============")
    
    cmd = [
        sys.executable,
        str(PROJ_ROOT / "run_trainval.py"),
        "--train_module", args.train_module,
        "--train_name", args.train_name,
        "--scene", args.scene,
        "--dataset_entry", f"'{args.dataset_entry}'",
        "--train_sub", str(args.train_sub),
        "--data_root", args.data_root,
        "--stage", str(stage)
    ]
    
    result = subprocess.run(" ".join(map(str, cmd)), shell=True, check=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Run training scripts for pose and depth optimization.')
    
    # Training configuration
    parser.add_argument('--train_module', 
                       type=str,
                       default="joint_pose_nerf_training/scannet_depth_exp",
                       help='Name of module in the "train_settings/" folder')
    parser.add_argument('--train_name',
                       type=str,
                       default="zoedepth_pdcnet",
                       help='Name of the train settings file')
    parser.add_argument('--train_sub',
                       type=int,
                       default=5,
                       help='Number of input views to consider')
    
    # Dataset configuration
    parser.add_argument('--data_root',
                       type=str,
                       default='/home/ubuntu/disk6/RSfM-Datasets/ScanNet',
                       help='Root directory of the dataset')
    parser.add_argument('--dataset',
                       type=str,
                       choices=["scannet", "kitti360"],
                       default="scannet",
                       help='Dataset type')
    parser.add_argument('--scene',
                       type=str,
                       default=None,
                       help='Scene identifier')
    parser.add_argument('--dataset_entry',
                       type=str,
                       default=None,
                       help='Dataset entry information')
    
    return parser.parse_args()

def main() -> None:
    """Main execution function for pose and depth optimization pipeline."""
    args = parse_args()

    while True:
        sequence_id, dataset_entry = select_next_sequence(args)
        if sequence_id is None:
            break
            
        print(f"Processing scene {sequence_id}\nEntry: {dataset_entry}")
        args.scene = sequence_id
        args.dataset_entry = dataset_entry

        for stage in [1, 2]:  # Stage 1: Pose optimization, Stage 2: Depth optimization
            run_optimization_stage(args, stage)

if __name__ == '__main__':
    main()
