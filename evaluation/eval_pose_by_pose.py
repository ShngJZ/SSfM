import glob
import os
import sys
import pickle
import argparse
import importlib
from typing import List, Tuple, Dict, Any

import kornia.geometry
import numpy as np
import torch
import torch.backends.cudnn
from easydict import EasyDict as edict
from tabulate import tabulate
from tqdm import tqdm

# Add project root to path
proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

import source.admin.settings as ws_settings
from analysis.utils_evaldepth import compute_depth_errors
from source.training.define_trainer import define_trainer
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.joint_pose_nerf_trainer import CommonPoseEvaluation
from source.training.core.triangulation_loss import padding_pose

def projection2corres(intr: torch.Tensor, poses: torch.Tensor, depthmap: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Project points from one view to another using camera poses.
    
    Args:
        intr: Camera intrinsic matrices
        poses: Camera poses (world to camera)
        depthmap: Depth maps
        
    Returns:
        Dictionary mapping frame pairs to correspondences
    """
    nfrm, H, W = depthmap.shape
    gridxx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    gridyy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    gridxx = gridxx.view(1, H, W, 1)
    gridyy = gridyy.view(1, H, W, 1)
    gridpxls = torch.cat([gridxx, gridyy], dim=-1).float().cuda()

    intr, poses, depthmap = padding_pose(intr).cuda(), padding_pose(poses).cuda(), depthmap.cuda()
    selector = depthmap > 0
    corres = {}
    
    for i in range(nfrm):
        for j in range(nfrm):
            if i == j:
                continue
            prjM = intr[j] @ poses[j] @ poses[i].inverse() @ intr[i].inverse()
            pts_source = gridpxls[0, selector[i], :]
            depthf = depthmap[i][selector[i]]
            pts_source_3D = kornia.geometry.convert_points_to_homogeneous(
                kornia.geometry.convert_points_to_homogeneous(pts_source) * depthf.unsqueeze(1)
            )
            pts_target_2D = pts_source_3D @ prjM.transpose(-1, -2)
            pts_target_2D = kornia.geometry.convert_points_from_homogeneous(
                kornia.geometry.convert_points_from_homogeneous(pts_target_2D)
            )
            corres[str([i, j])] = pts_target_2D
            
    return corres

def compute_depth_errors_batch(depth_gt: torch.Tensor, depth_est: torch.Tensor) -> np.ndarray:
    """Compute depth errors for a batch of depth maps."""
    metrics = [compute_depth_errors(depth_gt[i].flatten(), depth_est[i].flatten()) 
              for i in range(len(depth_gt))]
    return np.stack(metrics, axis=0)

def evaluate_depthmap_scale_improvement(pose_net, data_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate improvement in depth map scale estimation."""
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
        gt_scale = np.median(depth_gt[i][selector]) / np.median(depth_est[i][selector])
        depth_upperbound.append(depth_est[i] * gt_scale)
    depth_upperbound = np.stack(depth_upperbound, axis=0)
    depth_metrics_upperbound = compute_depth_errors_batch(depth_gt[1:], depth_upperbound)
    
    return depth_metrics_est[:, 0:9], depth_metrics_opt[:, 0:9], depth_metrics_upperbound[:, 0:9]

def init_settings(train_module: str, train_name: str, args: Dict[str, Any]) -> edict:
    """Initialize settings for evaluation."""
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

def pose_auc(errors: np.ndarray, thresholds: List[float]) -> List[float]:
    """Compute area under curve (AUC) for pose errors at different thresholds."""
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs

def evaluate_camera_relative(opt: edict, 
                           pose_w2c: torch.Tensor, 
                           pose_GT_w2c: torch.Tensor) -> Tuple[np.ndarray, float, float]:
    """Evaluate relative camera pose errors."""
    def compute_relposes(poses: torch.Tensor) -> torch.Tensor:
        """Compute relative poses with respect to first frame."""
        from source.utils.camera import pad_poses, unpad_poses
        relposes = []
        for i in range(len(poses)):
            relpose = unpad_poses(pad_poses(poses[0]) @ pad_poses(poses[i]).inverse())
            relpose[:, 3] = relpose[:, 3] / (relpose[:, 3] ** 2 + 1e-10).sum().sqrt()
            relposes.append(relpose)
        return torch.stack(relposes, dim=0)

    def compute_relpose_error_deg(T_0to1_gt: torch.Tensor, T_0to1_est: torch.Tensor) -> Tuple[float, float]:
        """Compute relative pose error in degrees."""
        def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
            cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
            cos = np.clip(cos, -1.0, 1.0)
            return np.rad2deg(np.abs(np.arccos(cos))).item()

        def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
            n = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

        R_gt = T_0to1_gt[:3, :3].cpu().numpy()
        t_gt = T_0to1_gt[:3, 3].cpu().numpy()
        R_est = T_0to1_est[:3, :3].cpu().numpy()
        t_est = T_0to1_est[:3, 3].cpu().numpy()

        error_t = angle_error_vec(t_est.squeeze(), t_gt)
        error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
        error_t = 180.0 if np.isnan(error_t) or np.isinf(error_t) else error_t

        error_R = angle_error_mat(R_est, R_gt)
        return error_t, error_R

    # Compute relative poses
    relpose = compute_relposes(pose_w2c)
    relpose_GT = compute_relposes(pose_GT_w2c)

    # Compute errors
    error_pose = []
    for i in range(1, len(pose_w2c)):
        error_t, error_R = compute_relpose_error_deg(relpose_GT[i], relpose[i])
        error_pose.append(max(error_t, error_R))
    error_pose = np.array(error_pose)

    # Align poses and compute errors
    evaluator = CommonPoseEvaluation()
    evaluator.settings = opt
    pose_aligned, _ = evaluator.prealign_w2c_small_camera_systems(opt, pose_w2c, pose_GT_w2c)
    error = evaluator.evaluate_camera_alignment(opt, pose_aligned, pose_GT_w2c)
    error_t = error['t'].cpu().numpy() * 100
    error_R = np.rad2deg(error['R'].cpu().numpy())
    
    return error_pose, error_t, error_R


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

    # Load dataset split
    split_export_path = os.path.join(proj_root, 'split', 'scannet', '{}.txt'.format(args.dataset))
    with open(split_export_path) as file:
        entries = file.readlines()

    init_deg_errs, init_t_errs, init_R_errs = [], [], []
    opt_deg_errs, opt_t_errs, opt_R_errs = [], [], []
    seqcnt, failure = 0, 0

    # Evaluate Pose
    for entry in tqdm(entries):
        combs = entry.rstrip('\n').split(' ')
        seq, rgbroot = combs[0], combs[1]
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
        pose_opted_path = os.path.join(project_path, "{}.pth".format(args.pose_eval))
        pose_init_path = os.path.join(project_path, "init_pose.pickle")
        input_data_path = os.path.join(project_path, "input_data.pickle")
        
        if not all(os.path.exists(p) for p in [pose_opted_path, pose_init_path, input_data_path]):
            continue

        # Load pose data
        with open(pose_init_path, 'rb') as f:
            pose_init = pickle.load(f)

        pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            settings, nbr_poses=args.train_sub, initial_poses_w2c=pose_init, device=torch.device("cpu")
        )
        pose_net.load_state_dict(torch.load(pose_opted_path), strict=True)
        pose_opted = pose_net.get_w2c_poses()

        # Load input data
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)

        if torch.isnan(input_data['pose']).any():
            continue

        # Evaluate pose estimation
        gt_poses_w2c = input_data['pose']
        init_deg_err, init_t_err, init_R_err = evaluate_camera_relative(settings, pose_init, gt_poses_w2c)
        opt_deg_err, opt_t_err, opt_R_err = evaluate_camera_relative(settings, pose_opted, gt_poses_w2c)

        init_deg_errs.append(init_deg_err)
        init_t_errs.append(init_t_err)
        init_R_errs.append(init_R_err)
        opt_deg_errs.append(opt_deg_err)
        opt_t_errs.append(opt_t_err)
        opt_R_errs.append(opt_R_err)

        seqcnt += 1

    # Compute final metrics
    error_pose_all = {
        'init': pose_auc(np.concatenate(init_deg_errs), [5, 10, 20]) + [
            np.concatenate(init_t_errs).mean(),
            np.concatenate(init_R_errs).mean()
        ],
        'opt': pose_auc(np.concatenate(opt_deg_errs), [5, 10, 20]) + [
            np.concatenate(opt_t_errs).mean(),
            np.concatenate(opt_R_errs).mean()
        ]
    }

    # Print results
    print("Performance of %s on %d / %d Seq, Success Rate %.1f" % (
        args.train_name, seqcnt, (seqcnt + failure), seqcnt / (seqcnt + failure) * 100
    ))

    table = [['', 'Auc-5', 'Auc-10', 'Auc-20', 'Error_t', 'Error_R']]
    for key, values in error_pose_all.items():
        table.append([key] + values)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

if __name__ == '__main__':
    main()
