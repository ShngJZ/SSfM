import glob
import os
import sys
import pickle
import argparse
import importlib
import shutil

import kornia.geometry
import matplotlib.pyplot as plt
import natsort
import numpy as np
import tqdm

proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

import torch
import torch.backends.cudnn
from easydict import EasyDict as edict
from tabulate import tabulate

import source.admin.settings as ws_settings
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.core.triangulation_loss import padding_pose
from analysis.utils_vls import tensor2disp

def init_settings(train_module: str, train_name: str, args: argparse.Namespace) -> edict:
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

    expr_module = importlib.import_module(
        'train_settings.{}.{}'.format(train_module_for_launching.replace('/', '.'), train_name.replace('/', '.'))
    )
    expr_func = getattr(expr_module, 'get_config')

    model_config = expr_func()
    settings.update(model_config)
    settings.train_sub = args.train_sub
    return settings

def to_cuda(bundle):
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

def produce_projective_correspondence(pose_i2j, intrinsic, depthmap):
    """Compute projective correspondences between two views."""
    H, W = depthmap.shape
    device = depthmap.device
    
    prjm_i2j = intrinsic @ pose_i2j @ intrinsic.inverse()
    prjm_i2j_3D = pose_i2j @ intrinsic.inverse()

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
    pts3d_prj = prjm_i2j_3D.view([1, 1, 4, 4]) @ pts3d.view([1, H, W, 4, 1])

    pts2d_prjx, pts2d_prjy = torch.split(pts2d_prj, 1, dim=-1)
    pts2d_prjx, pts2d_prjy = pts2d_prjx.view([H, W]), pts2d_prjy.view([H, W])

    corres = torch.cat([pts2d, pts2d_prj], dim=-1)
    valid = (depthmap > 0) * (pts2d_prjx > 0) * (pts2d_prjx < W - 1) * (pts2d_prjy > 0) * (pts2d_prjy < H - 1)
    return corres, valid, pts3d_prj

def produce_projective_correspondence_from_poses(intrinsic, gt_poses_w2c, depth_gt, flow_pairs):
    """Compute projective correspondences for multiple pose pairs."""
    _, H, W = depth_gt.shape
    device = depth_gt.device
    valid_gt_th002_pairs, valid_gt_th003_pairs, valid_gt_th005_pairs = list(), list(), list()
    
    for idx, (frmi, frmj) in enumerate(flow_pairs):
        depth_gt_frmi, depth_gt_frmj = depth_gt[frmi], depth_gt[frmj]
        intrinsic_i, pose_i2j = intrinsic[frmi], gt_poses_w2c[frmj] @ gt_poses_w2c[frmi].inverse()
        corres_i2j, valid_i2j, pts3d_viewi2j = produce_projective_correspondence(pose_i2j, intrinsic_i, depth_gt_frmi)

        # Apply Consistency Check
        source_grid, target_grid = torch.split(corres_i2j, 2, dim=-1)
        target_gridx, target_gridy = torch.split(target_grid, 1, dim=-1)
        target_gridx = (target_gridx / (W - 1) - 0.5) * 2
        target_gridy = (target_gridy / (H - 1) - 0.5) * 2
        sample_grid = torch.cat([target_gridx, target_gridy], dim=-1).view([1, H, W, 2])

        depth_gt_recon = torch.nn.functional.grid_sample(depth_gt_frmj.view([1, 1, H, W]), sample_grid, align_corners=True, padding_mode="zeros")
        depth_gt_recon = depth_gt_recon.view([H, W])

        pts3d_viewi2j_sampled = kornia.geometry.convert_points_to_homogeneous(target_grid) * depth_gt_recon.view([H, W, 1])
        pts3d_viewi2j_sampled = kornia.geometry.convert_points_to_homogeneous(pts3d_viewi2j_sampled)
        pts3d_viewi2j_sampled = intrinsic_i.inverse().view([1, 1, 4, 4]) @ pts3d_viewi2j_sampled.view([H, W, 4, 1])

        pts3d_viewi2j, pts3d_viewi2j_sampled = pts3d_viewi2j.view([H, W, 4]), pts3d_viewi2j_sampled.view([H, W, 4])
        dist = kornia.geometry.convert_points_to_homogeneous(pts3d_viewi2j) - kornia.geometry.convert_points_to_homogeneous(pts3d_viewi2j_sampled)
        dist = torch.sqrt(torch.sum(dist ** 2, dim=-1) + 1e-10)
        
        valid_gt_th002 = (dist < 0.01) * valid_i2j
        valid_gt_th003 = (dist < 0.03) * valid_i2j
        valid_gt_th005 = (dist < 0.05) * valid_i2j

        valid_gt_th002_pairs.append(valid_gt_th002)
        valid_gt_th003_pairs.append(valid_gt_th003)
        valid_gt_th005_pairs.append(valid_gt_th005)
        
    valid_gt_th002_pairs = torch.stack(valid_gt_th002_pairs, dim=0).view([-1, 1, H, W])
    valid_gt_th003_pairs = torch.stack(valid_gt_th003_pairs, dim=0).view([-1, 1, H, W])
    valid_gt_th005_pairs = torch.stack(valid_gt_th005_pairs, dim=0).view([-1, 1, H, W])
    return valid_gt_th002_pairs, valid_gt_th003_pairs, valid_gt_th005_pairs

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
    parser.add_argument('--split', type=str, default="scannet")
    args = parser.parse_args()

    split_export_path = os.path.join(proj_root, 'split', 'scannet', '{}.txt'.format(args.split))
    with open(split_export_path) as file:
        entries = file.readlines()

    metric_init_all, metric_opted_all, seqcnt, failure = list(), list(), 0, 0
    
    # Evaluate Pose
    for entry in tqdm.tqdm(entries):
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

        project_path = os.path.join(proj_root, 'checkpoint', train_module, args.train_name)
        pose_opted = os.path.join(project_path, "{}.pth".format(args.pose_eval))
        if not os.path.exists(pose_opted):
            continue
            
        pose_init = os.path.join(project_path, "init_pose.pickle")
        pose_init = pickle.load(open(pose_init, 'rb'))

        pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            settings, nbr_poses=args.train_sub, initial_poses_w2c=pose_init, device=torch.device("cpu")
        )

        pose_net.load_state_dict(torch.load(pose_opted), strict=True)
        pose_opted = pose_net.get_w2c_poses()

        input_data_path = os.path.join(project_path, "input_data.pickle")
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)
        input_data = to_cuda(input_data)

        if torch.sum(torch.isnan(input_data['pose'])) > 0:
            continue

        correspondence_path = os.path.join(project_path, "correspondence.pth")
        correspondence_bundle = torch.load(correspondence_path)
        correspondence_bundle = to_cuda(correspondence_bundle)
        corres_maps, flow_pairs, conf_maps, mask_valid_corr = correspondence_bundle

        intrinsic, gt_poses_w2c = padding_pose(input_data['intr']), padding_pose(input_data['pose'])
        depth_gt = input_data['depth_gt']

        valid_gt_the001, valid_gt_the003, valid_gt_the005 = produce_projective_correspondence_from_poses(
            intrinsic, gt_poses_w2c, depth_gt, flow_pairs
        )
        valid_init_the001, valid_init_the003, valid_init_the005 = produce_projective_correspondence_from_poses(
            intrinsic, padding_pose(pose_init).cuda(), depth_gt, flow_pairs
        )
        valid_opted_the001, valid_opted_the003, valid_opted_the005 = produce_projective_correspondence_from_poses(
            intrinsic, padding_pose(pose_opted).cuda(), depth_gt, flow_pairs
        )

        # Calculate metrics
        a1_init = (valid_init_the001 * valid_gt_the001).sum() / valid_gt_the001.sum()
        a2_init = (valid_init_the003 * valid_gt_the003).sum() / valid_gt_the003.sum()
        a3_init = (valid_init_the005 * valid_gt_the005).sum() / valid_gt_the005.sum()

        a1_opted = (valid_opted_the001 * valid_gt_the001).sum() / valid_gt_the001.sum()
        a2_opted = (valid_opted_the003 * valid_gt_the003).sum() / valid_gt_the003.sum()
        a3_opted = (valid_opted_the005 * valid_gt_the005).sum() / valid_gt_the005.sum()

        metric_init = [a1_init.item(), a2_init.item(), a3_init.item()]
        metric_opted = [a1_opted.item(), a2_opted.item(), a3_opted.item()]
        metric_init_all.append(np.array(metric_init))
        metric_opted_all.append(np.array(metric_opted))

        seqcnt += 1

    # Compute final metrics
    metric_init_all = np.mean(np.stack(metric_init_all, axis=0), axis=0)
    metric_opted_all = np.mean(np.stack(metric_opted_all, axis=0), axis=0)
    metrics = {
        'Init': metric_init_all,
        'Optimized': metric_opted_all,
    }

    # Print results
    print("Performance of %s on %d / %d Seq, Success Rate %.1f" % (
        args.train_name, seqcnt, (seqcnt + failure), seqcnt / (seqcnt + failure) * 100
    ))

    table = [['', 'c3D01', 'c3D03', 'c3D05']]
    for key in metrics.keys():
        table_entry = [key] + list(metrics[key])
        table.append(table_entry)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

if __name__ == '__main__':
    main()
