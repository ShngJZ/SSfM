import glob
import natsort
import os
import sys
import pickle
import argparse
import importlib
from typing import List, Tuple, Dict, Any

import kornia.geometry
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from easydict import EasyDict as edict
from tabulate import tabulate
from tqdm import tqdm

# Add project root to path
proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

import source.admin.settings as ws_settings
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.core.triangulation_loss import padding_pose
from analysis.utils_vls import tensor2disp

def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute least squares solution parameters of Sim(m) matrix for point pattern alignment.
    
    Implementation of the paper: Umeyama, Shinji: Least-squares estimation of transformation parameters
    between two point patterns. IEEE PAMI, 1991
    
    Args:
        x: mxn matrix of points, m = dimension, n = nr. of data points
        y: mxn matrix of points, m = dimension, n = nr. of data points
        with_scale: Set to True to align also the scale (default: True)
        
    Returns:
        Tuple containing:
        - r: Rotation matrix
        - t: Translation vector
        - c: Scale factor
    """
    if x.shape != y.shape:
        raise ValueError("x.shape not equal to y.shape")

    m, n = x.shape  # m = dimension, n = nr. of data points

    # Compute means (eq. 34 and 35)
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # Compute variance (eq. 36)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # Compute covariance matrix (eq. 38)
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text between eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix (eq. 43)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1  # Ensure RHS coordinate system

    # Compute rotation (eq. 40)
    r = u.dot(s).dot(v)

    # Compute scale and translation (eq. 42 and 41)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

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

def to_cuda(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Move tensors to CUDA device."""
    if isinstance(bundle, dict):
        for x in bundle.keys():
            if isinstance(bundle[x], torch.Tensor):
                bundle[x] = bundle[x].cuda()
    elif isinstance(bundle, list):
        for i in range(len(bundle)):
            if isinstance(bundle[i], torch.Tensor):
                bundle[i] = bundle[i].cuda()
    return bundle

def produce_projective_correspondence(pose_i2j: torch.Tensor, 
                                   intrinsic: torch.Tensor, 
                                   depthmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute projective correspondences between two views."""
    H, W = depthmap.shape
    device = depthmap.device
    prjm_i2j = intrinsic @ pose_i2j @ intrinsic.inverse()

    # Create pixel coordinate grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(H, W).float().to(device)
    yy = yy.view(H, W).float().to(device)

    pts2d = torch.stack([xx, yy], dim=-1)
    pts3d = kornia.geometry.convert_points_to_homogeneous(pts2d) * depthmap.view([H, W, 1])
    pts3d = kornia.geometry.convert_points_to_homogeneous(pts3d)
    pts2d_prj = prjm_i2j.view([1, 1, 4, 4]) @ pts3d.view([1, H, W, 4, 1])
    pts2d_prj = pts2d_prj.view([H, W, 4])
    pts2d_prj = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts2d_prj))

    pts2d_prjx, pts2d_prjy = torch.split(pts2d_prj, 1, dim=-1)
    pts2d_prjx, pts2d_prjy = pts2d_prjx.view([H, W]), pts2d_prjy.view([H, W])

    corres = torch.cat([pts2d, pts2d_prj], dim=-1)
    valid = (depthmap > 0) * (pts2d_prjx > 0) * (pts2d_prjx < W - 1) * (pts2d_prjy > 0) * (pts2d_prjy < H - 1)
    return corres, valid

def produce_projective_correspondence_from_poses(intrinsic: torch.Tensor,
                                              gt_poses_w2c: torch.Tensor,
                                              depth_gt: torch.Tensor,
                                              flow_pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute projective correspondences for multiple pose pairs."""
    _, H, W = depth_gt.shape
    device = depth_gt.device
    corres_i2j_pairs, valid_i2j_pairs = [], []
    
    for idx, (frmi, frmj) in enumerate(flow_pairs):
        depth_gt_frmi, depth_gt_frmj = depth_gt[frmi], depth_gt[frmj]
        intrinsic_i, pose_i2j = intrinsic[frmi], gt_poses_w2c[frmj] @ gt_poses_w2c[frmi].inverse()
        corres_i2j, valid_i2j = produce_projective_correspondence(pose_i2j, intrinsic_i, depth_gt_frmi)

        # Apply Consistency Check
        source_grid, target_grid = torch.split(corres_i2j, 2, dim=-1)
        target_gridx, target_gridy = torch.split(target_grid, 1, dim=-1)
        target_gridx = (target_gridx / (W - 1) - 0.5) * 2
        target_gridy = (target_gridy / (H - 1) - 0.5) * 2
        sample_grid = torch.cat([target_gridx, target_gridy], dim=-1).view([1, H, W, 2])

        depth_gt_recon = F.grid_sample(depth_gt_frmj.view([1, 1, H, W]), sample_grid, align_corners=True, padding_mode="zeros")
        depth_gt_recon = depth_gt_recon.view([H, W])

        # Compute consistency mask
        pts3d_viewi2j = kornia.geometry.convert_points_to_homogeneous(target_grid) * depth_gt_recon.view([H, W, 1])
        pts3d_viewi2j = kornia.geometry.convert_points_to_homogeneous(pts3d_viewi2j)
        prjM_i2j2i = intrinsic_i @ pose_i2j.inverse() @ intrinsic_i.inverse()
        pts3d_viewi2j2i = prjM_i2j2i.view([1, 1, 4, 4]) @ pts3d_viewi2j.view([H, W, 4, 1])
        pts3d_viewi2j2i = pts3d_viewi2j2i.view([H, W, 4])
        pts3d_viewi2j2i = kornia.geometry.convert_points_from_homogeneous(pts3d_viewi2j2i)
        _, _, depth_consis_ck = torch.split(pts3d_viewi2j2i, 1, dim=-1)
        depth_consis_ck = depth_consis_ck.view([H, W])

        consist_mask = ((depth_consis_ck - depth_gt_frmi) / depth_gt_frmi).abs() < 0.05
        valid_i2j = valid_i2j * consist_mask

        corres_i2j_pairs.append(corres_i2j[:, :, 2:4])
        valid_i2j_pairs.append(valid_i2j)
        
    corres_i2j_pairs = torch.stack(corres_i2j_pairs, dim=0).permute([0, 3, 1, 2]).contiguous()
    valid_i2j_pairs = torch.stack(valid_i2j_pairs, dim=0).view([-1, 1, H, W])
    return corres_i2j_pairs, valid_i2j_pairs

def compute_corres_error(corres_est: torch.Tensor, 
                        corres_gt: torch.Tensor, 
                        valid_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute correspondence error metrics."""
    epe = torch.sqrt(torch.sum((corres_est - corres_gt) ** 2, dim=1, keepdim=True) + 1e-12)
    epes = epe[valid_gt]
    px1s, px3s, px5s = (epes < 1), (epes < 3), (epes < 5)
    metric = torch.Tensor([epes.mean(), px1s.float().mean(), px3s.float().mean(), px5s.float().mean()])
    return epes, px1s, px3s, px5s, metric


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

    metric_bs_all, metric_init_all, metric_opted_all, pxl_all = [], [], [], []
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
        pose_init_path = os.path.join(project_path, "init_pose.pickle")
        pose_opted_path = os.path.join(project_path, "{}.pth".format(args.pose_eval))
        
        if not all(os.path.exists(p) for p in [project_path, pose_init_path, pose_opted_path]):
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
        input_data_path = os.path.join(project_path, "input_data.pickle")
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)
        input_data = to_cuda(input_data)

        # Load correspondence data
        correspondence_path = os.path.join(project_path, "correspondence.pth")
        correspondence_bundle = torch.load(correspondence_path)
        correspondence_bundle = to_cuda(correspondence_bundle)
        corres_maps, flow_pairs, conf_maps, mask_valid_corr = correspondence_bundle

        # Process data
        intrinsic = padding_pose(input_data['intr'])
        gt_poses_w2c = padding_pose(input_data['pose'])
        depth_gt = input_data['depth_gt']
        depth_est = input_data['depth_est']

        # Compute correspondences
        corres_gt, valid_gt = produce_projective_correspondence_from_poses(
            intrinsic, gt_poses_w2c, depth_gt, flow_pairs
        )
        corres_init, _ = produce_projective_correspondence_from_poses(
            intrinsic, padding_pose(pose_init).cuda(), depth_gt, flow_pairs
        )
        corres_opted, _ = produce_projective_correspondence_from_poses(
            intrinsic, padding_pose(pose_opted).cuda(), depth_gt, flow_pairs
        )

        # Compute metrics
        confs = conf_maps[valid_gt]
        epes_bs, px1s_bs, px3s_bs, px5s_bs, metric_bs = compute_corres_error(corres_maps, corres_gt, valid_gt)
        epes_int, px1s_init, px3s_init, px5s_init, metric_init = compute_corres_error(corres_init, corres_gt, valid_gt)
        epes_opted, px1s_opted, px3s_opted, px5s_opted, metric_opted = compute_corres_error(corres_opted, corres_gt, valid_gt)

        pxl = torch.stack([px1s_bs.float().cpu(), px1s_init.float().cpu(), px1s_opted.float().cpu(), confs.cpu()], dim=-1)
        pxl = pxl[::100]
        pxl_all.append(pxl)

        metric_bs_all.append(metric_bs)
        metric_init_all.append(metric_init)
        metric_opted_all.append(metric_opted)
        seqcnt += 1

    # Compute final metrics
    metric_bs_all = torch.mean(torch.stack(metric_bs_all, dim=0), dim=0).cpu().numpy()
    metric_init_all = torch.mean(torch.stack(metric_init_all, dim=0), dim=0).cpu().numpy()
    metric_opted_all = torch.mean(torch.stack(metric_opted_all, dim=0), dim=0).cpu().numpy()
    metrics = {
        'Corres': metric_bs_all,
        'Init': metric_init_all,
        'Optimized': metric_opted_all,
    }

    # Print results
    print("Performance of %s on %d / %d Seq, Success Rate %.1f" % (
        args.train_name, seqcnt, (seqcnt + failure), seqcnt / (seqcnt + failure) * 100
    ))

    table = [['', 'epe', 'px1', 'px3', 'px5']]
    for key, values in metrics.items():
        table.append([key] + list(values))
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

if __name__ == '__main__':
    main()
