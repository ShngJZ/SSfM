import glob
import os
import sys
import pickle
import argparse
import importlib
from typing import Any, Dict, List, Tuple

import kornia.geometry
import natsort
import numpy as np
import torch
import torch.backends.cudnn
import cv2 as cv
from easydict import EasyDict as edict
from tabulate import tabulate
from tqdm import tqdm

# Add project root to path
proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

import source.admin.settings as ws_settings
from analysis.utils_evaldepth import compute_depth_errors
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters

def compute_depth_errors_batch(depth_gt: torch.Tensor, depth_est: torch.Tensor) -> np.ndarray:
    """Compute depth errors for a batch of depth maps.
    
    Args:
        depth_gt: Ground truth depth maps
        depth_est: Estimated depth maps
        
    Returns:
        Array of depth error metrics for each pair of depth maps
    """
    metrics = [compute_depth_errors(depth_gt[i].flatten(), depth_est[i].flatten()) 
              for i in range(len(depth_gt))]
    return np.stack(metrics, axis=0)

def evaluate_depthmap_scale_improvement(pose_net, data_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate improvement in depth map scale estimation.
    
    Args:
        pose_net: Network for pose and depth optimization
        data_dict: Dictionary containing depth maps and ground truth
        
    Returns:
        Tuple of (original depth metrics, optimized depth metrics, upperbound depth metrics)
    """
    depth_est = data_dict['depth_est']
    depth_optimized = pose_net.optimize_depth(depth_est)
    depth_gt = data_dict['depth_gt']

    # Compute metrics for original and optimized depth
    depth_metrics_est = compute_depth_errors_batch(depth_gt[1:], depth_est[1:])
    depth_metrics_opt = compute_depth_errors_batch(depth_gt[1:], depth_optimized[1:])

    # Compute upperbound metrics using ground truth scale
    depth_upperbound = []
    for i in range(1, len(depth_gt)):
        selector = depth_gt[i] > 0
        gt_scale = torch.median(depth_gt[i][selector]) / torch.median(depth_est[i][selector])
        depth_upperbound.append(depth_est[i] * gt_scale)
    depth_upperbound = torch.stack(depth_upperbound, axis=0)
    depth_metrics_upperbound = compute_depth_errors_batch(depth_gt[1:], depth_upperbound)
    
    return depth_metrics_est[:, 0:9], depth_metrics_opt[:, 0:9], depth_metrics_upperbound[:, 0:9]

def init_settings(train_module: str, train_name: str, args: Dict[str, Any]) -> edict:
    """Initialize settings for evaluation.
    
    Args:
        train_module: Name of the training module
        train_name: Name of the training configuration
        args: Command line arguments
        
    Returns:
        Settings dictionary with configuration
    """
    settings = ws_settings.Settings(data_root='')
    train_module_for_launching = train_module

    # Update module path with subset and scene information
    base_dir_train_module = train_module.split('/')
    if args.train_sub is not None and args.train_sub != 0:
        base_dir_train_module[1] += '/subset_' + str(args.train_sub)
    else:
        args.train_sub = None

    if args.scene is not None:
        base_dir_train_module[1] += '/' + args.scene
    train_module = '/'.join(base_dir_train_module)

    settings.module_name_for_eval = train_module_for_launching
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = '{}/{}'.format(train_module, train_name)

    settings = edict(settings.__dict__)

    # Import and get configuration
    expr_module = importlib.import_module(
        'train_settings.{}.{}'.format(train_module_for_launching.replace('/', '.'), train_name.replace('/', '.'))
    )
    model_config = getattr(expr_module, 'get_config')()

    settings.update(model_config)
    settings.train_sub = args.train_sub
    return settings

def to_cuda(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Move tensors to CUDA device.
    
    Args:
        bundle: Dictionary or list containing tensors
        
    Returns:
        Bundle with tensors moved to CUDA
    """
    if isinstance(bundle, dict):
        for x in bundle.keys():
            if isinstance(bundle[x], torch.Tensor):
                bundle[x] = bundle[x].cuda()
    elif isinstance(bundle, list):
        for i in range(len(bundle)):
            if isinstance(bundle[i], torch.Tensor):
                bundle[i] = bundle[i].cuda()
    return bundle

@torch.no_grad()
def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, default="joint_pose_nerf_training/scannet_depth_exp",
                      help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default="zoedepth_pdcnet",
                      help='Name of the train settings file.')
    parser.add_argument('--train_sub', type=int, default=5,
                        help='train subset: how many input views to consider?')
    parser.add_argument('--pose_eval', type=str, help='Pose To Eval', default="pose_optimized")
    parser.add_argument('--dataset', type=str, default="scannet")
    args = parser.parse_args()

    assert args.dataset == "scannet"
    split_export_path = os.path.join(proj_root, 'split', 'scannet', 'scannet.txt')
    with open(split_export_path) as file:
        entries = file.readlines()

    error_org_depth, error_opt_depth, error_upperbound_depth = [], [], []
    seqcnt = 0

    # Evaluate Pose
    for entry in tqdm(entries):
        seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
        args.scene = seq
        settings = init_settings(args.train_module, args.train_name, args)
        train_module_for_launching, train_module = args.train_module, args.train_module

        # Update module path
        base_dir_train_module = train_module.split('/')
        if args.train_sub is not None and args.train_sub != 0:
            base_dir_train_module[1] += '/subset_' + str(args.train_sub)
        else:
            args.train_sub = None

        if args.scene is not None:
            base_dir_train_module[1] += '/' + args.scene
        train_module = '/'.join(base_dir_train_module)

        # Check required files exist
        project_path = os.path.join(proj_root, 'checkpoint', train_module, args.train_name)
        pose_init_path = os.path.join(project_path, "init_pose.pickle")
        pose_opted_path = os.path.join(project_path, "{}.pth".format(args.pose_eval))
        input_data_path = os.path.join(project_path, "input_data.pickle")
        
        if not all(os.path.exists(p) for p in [project_path, pose_init_path, pose_opted_path, input_data_path]):
            continue

        # Load pose data
        with open(pose_init_path, 'rb') as f:
            pose_init = pickle.load(f)

        pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            settings, nbr_poses=args.train_sub, initial_poses_w2c=pose_init, device=torch.device("cpu")
        )
        pose_net.load_state_dict(torch.load(pose_opted_path), strict=True)
        pose_net = pose_net.cuda()

        # Load input data
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)
        input_data = to_cuda(input_data)

        # Evaluate Depth
        depth_metrics_est, depth_metrics_opt, depth_metrics_upperbound = evaluate_depthmap_scale_improvement(
            pose_net, input_data
        )

        error_opt_depth.append(depth_metrics_opt)
        error_org_depth.append(depth_metrics_est)
        error_upperbound_depth.append(depth_metrics_upperbound)
        seqcnt += 1

    # Compute final metrics
    error_opt_depth = np.concatenate(error_opt_depth, axis=0)
    error_org_depth = np.concatenate(error_org_depth, axis=0)
    error_upperbound_depth = np.concatenate(error_upperbound_depth, axis=0)

    error_depth_all = {
        'org_mono': list(np.mean(error_org_depth, axis=0)),
        'Uppperbound': list(np.mean(error_upperbound_depth, axis=0)),
        'Opted': list(np.mean(error_opt_depth, axis=0))
    }

    # Print results
    table = [['', 'sc_inv', 'log10', 'silog', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd05', 'd1']]
    for key, values in error_depth_all.items():
        table.append([key] + values)

    print("=====Evaluated on %d Seqs====" % seqcnt)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

if __name__ == '__main__':
    main()
