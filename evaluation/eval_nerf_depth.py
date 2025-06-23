import glob
import os
import sys
import pickle
import argparse
import importlib
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm
import natsort
from easydict import EasyDict as edict
from tabulate import tabulate

proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

import source.admin.settings as ws_settings
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.core.triangulation_loss import padding_pose
from source.training.joint_pose_nerf_trainer import Graph
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_backproject_to_3d, batch_project
from analysis.utils_evaldepth import compute_depth_errors

def init_settings(train_module: str, train_name: str, args: argparse.Namespace) -> edict:
    """Initialize evaluation settings.

    Args:
        train_module: Name of training module
        train_name: Name of training configuration
        args: Command line arguments

    Returns:
        Settings dictionary with configuration parameters
    """
    settings = ws_settings.Settings(data_root='')
    train_module_for_launching = train_module

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

def to_cuda(bundle: Any) -> Any:
    """Move data to CUDA device.

    Args:
        bundle: Input data (dict or list containing tensors)

    Returns:
        Data with tensors moved to CUDA
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
def acquire_pts3D_depth(data_dict: Dict, nerf: Graph, settings: edict, rendered: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Acquire 3D points and depth maps.

    Args:
        data_dict: Input data dictionary
        nerf: NeRF model
        settings: Configuration settings
        rendered: Whether to use NeRF rendering or monocular depth

    Returns:
        Tuple of (3D points, depth maps)
    """
    B, _, H, W = data_dict['image'].shape
    id_ref, nbr = 0, int(settings.nerf.rand_rays * 10)

    depth_range = [1 / data_dict.depth_range[0, 0].item(), 1 / data_dict.depth_range[0, 1].item()]
    depth_range_in_self, depth_range_in_other = depth_range, depth_range

    poses_w2c = padding_pose(nerf.get_w2c_pose(settings, data_dict, mode='eval'))
    poses_c2w = pose_inverse_4x4(poses_w2c)
    intr = data_dict.intr

    intr_ref = intr[id_ref]
    pose_w2c_ref = poses_w2c[id_ref]
    pose_c2w_ref = poses_c2w[id_ref]

    # Sample Ray
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(H, W).float().cuda()
    yy = yy.view(H, W).float().cuda()
    pixels_ref = torch.stack([xx, yy], dim=-1).view([int(H*W), 2])
    nbr = len(pixels_ref)

    # Sample Reference Depthmap
    if rendered:
        depth_ref = nerf.render_image_at_specific_pose_and_rays(
            settings, data_dict, depth_range_in_self, pose_w2c_ref[:3], intr=intr_ref, H=H, W=W,
            pixels=pixels_ref, mode='val', iter=0).depth.view([nbr])
    else:
        depth_ref = data_dict['depth_est'][id_ref, pixels_ref[:, 1].long(), pixels_ref[:, 0].long()]

    pts3d_in_w_from_ref = batch_backproject_to_3d(kpi=pixels_ref, di=depth_ref, Ki=intr_ref, T_itoj=pose_c2w_ref)

    # Produce Sample Locations in other view
    sample_location = {0: pixels_ref}
    for k in range(1, settings.train_sub):
        pose_w2c_at_sampled = poses_w2c[k]
        intr_at_sampled = intr_ref.clone()
        pts_in_sampled_img, _ = batch_project(pts3d_in_w_from_ref,
                                            T_itoj=pose_w2c_at_sampled,
                                            Kj=intr_at_sampled,
                                            return_depth=True)
        sample_location[k] = pts_in_sampled_img

    # Render
    pts3D_rec, depth_rec = [pts3d_in_w_from_ref], [depth_ref]
    for k in range(1, settings.train_sub):
        pose_w2c_at_sampled = poses_w2c[k]
        pose_c2w_at_sampled = poses_c2w[k]
        intr_at_sampled = intr_ref.clone()
        pts_in_sampled_img = sample_location[k]
        if rendered:
            depth_ck = nerf.render_image_at_specific_pose_and_rays(
                settings, data_dict, depth_range_in_self, pose_w2c_at_sampled[:3],
                intr=intr_at_sampled, H=H, W=W, pixels=pts_in_sampled_img, mode='val', iter=0).depth.view([nbr])
        else:
            pxlx, pxly = torch.split(pts_in_sampled_img, 1, dim=1)
            pxlx, pxly = (pxlx / (W - 1) - 0.5) * 2, (pxly / (H - 1) - 0.5) * 2
            sample_grid = torch.cat([pxlx, pxly], dim=-1).view([1, 1, nbr, 2])
            depth_ck = torch.nn.functional.grid_sample(data_dict['depth_est'][k].view([1, 1, H, W]), sample_grid, align_corners=True, padding_mode="zeros").view([nbr])

        pts3d_in_w_from_ck = batch_backproject_to_3d(
            kpi=pts_in_sampled_img, di=depth_ck, Ki=intr_at_sampled, T_itoj=pose_c2w_at_sampled)
        pts3D_rec.append(pts3d_in_w_from_ck)
        depth_rec.append(depth_ck)

    pts3D_rec = torch.stack(pts3D_rec, dim=0).view([B, H, W, 3])
    depth_rec = torch.stack(depth_rec, dim=0).view([B, H, W, 1])
    return pts3D_rec, depth_rec

@torch.no_grad()
def analysis(data_dict: Dict, nerf: Graph, settings: edict, th_number: int = 2,
            relative_count: int = 1, dense: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze depth consistency between NeRF and monocular estimates.

    Args:
        data_dict: Input data dictionary
        nerf: NeRF model
        settings: Configuration settings
        th_number: Threshold for number of consistent views
        relative_count: Required difference in consistent view count
        dense: Whether to use dense evaluation

    Returns:
        Tuple of (monocular metrics, NeRF metrics, valid percentages)
    """
    pts3D_nerf, depth_nerf = acquire_pts3D_depth(data_dict, nerf, settings, rendered=True)
    pts3D_mono, depth_mono = acquire_pts3D_depth(data_dict, nerf, settings, rendered=False)

    B, H, W, _ = depth_nerf.shape

    def compute_normalized_3D_dist(pts3D: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Compute normalized 3D distances between reference and other views."""
        B = len(pts3D)
        pts3D_src, pts3D_sup = torch.split(pts3D, [1, B - 1], dim=0)
        depth_src, depth_sup = torch.split(depth, [1, B - 1], dim=0)
        pts3d_diff = torch.sqrt(torch.sum((pts3D_src - pts3D_sup) ** 2, dim=-1, keepdim=True) + 1e-10) / depth_sup
        return pts3d_diff

    pts3d_diff_nerf = compute_normalized_3D_dist(pts3D_nerf, depth_nerf)
    pts3d_diff_mono = compute_normalized_3D_dist(pts3D_mono, depth_mono)

    depth_mono_src = depth_mono[0]
    depth_nerf_src = depth_nerf[0]
    depth_gt_src = data_dict['depth_gt'][0].view([H, W, 1])

    th_dists = np.exp(np.linspace(np.log(0.05), np.log(0.0005), 200))
    mono_metrics, nerf_metrics, val_percent = [], [], []

    for th_dist in th_dists:
        val_observation = torch.sum(pts3d_diff_nerf < th_dist, dim=0) >= th_number
        val_observation = val_observation * ((torch.sum(pts3d_diff_nerf < th_dist, dim=0) - torch.sum(pts3d_diff_mono < th_dist, dim=0)) >= relative_count)

        if dense:
            val_observation = torch.zeros_like(val_observation) > -100000

        if torch.sum(val_observation) < 1000:
            val_observation = torch.ones_like(val_observation) == 0

        mono_metric = compute_depth_errors(depth_gt_src[val_observation], depth_mono_src[val_observation])
        nerf_metric = compute_depth_errors(depth_gt_src[val_observation], depth_nerf_src[val_observation])
        mono_metrics.append(np.array(mono_metric))
        nerf_metrics.append(np.array(nerf_metric))
        val_percent.append(torch.sum(val_observation).item() / int(H * W))

    return (np.stack(mono_metrics, axis=0),
            np.stack(nerf_metrics, axis=0),
            np.array(val_percent))

def average_metric(metrics: np.ndarray) -> torch.Tensor:
    """Average metrics while skipping NaN values.

    Args:
        metrics: Input metrics array

    Returns:
        Averaged metrics
    """
    nseq, naba, nmetric = metrics.shape
    metrics_tmp = torch.from_numpy(metrics).view([nseq, int(naba * nmetric)]).contiguous()

    metrics_mean = []
    metrics_tmp = metrics_tmp.numpy()
    for i in range(metrics_tmp.shape[1]):
        metrics_mean.append(metrics_tmp[:, i].mean())
    return torch.from_numpy(np.array(metrics_mean)).view([naba, nmetric])

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, default="joint_pose_nerf_training/scannet_depth_exp",
                      help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default="zoedepth_pdcnet",
                      help='Name of the train settings file.')
    parser.add_argument('--train_sub', type=int, default=5,
                      help='Train subset: how many input views to consider?')
    parser.add_argument('--dataset', type=str, default="scannet",
                      help='Dataset name')
    args = parser.parse_args()

    split_export_path = os.path.join(proj_root, 'split', 'scannet', 'scannet.txt')
    with open(split_export_path) as file:
        entries = file.readlines()

    seqcnt = 0
    mono_metrics_all, nerf_metrics_all, val_percent_all = [], [], []

    for entry in tqdm.tqdm(entries):
        seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
        args.scene = seq
        project_path = os.path.join(proj_root, 'checkpoint', args.train_module,
                                  'subset_{}'.format(str(args.train_sub)), args.scene, args.train_name)

        if not os.path.exists(project_path):
            continue

        nerf_ckpt_paths = glob.glob(os.path.join(project_path, '*.pth.tar'))
        nerf_ckpt_paths = natsort.natsorted(nerf_ckpt_paths)
        if len(nerf_ckpt_paths) == 0:
            continue

        nerf_ckpt_path = nerf_ckpt_paths[-1]
        nerf_ckpt_path_iter = int(nerf_ckpt_path.split('/')[-1].split('.')[0].split('-')[1])

        if int(nerf_ckpt_path_iter) < 80000:
            continue

        settings = init_settings(args.train_module, args.train_name, args)
        nerf_ckpt = torch.load(nerf_ckpt_path)['state_dict']

        pose_init = os.path.join(project_path, "init_pose.pickle")
        pose_init = pickle.load(open(pose_init, 'rb'))
        pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            settings, nbr_poses=args.train_sub, initial_poses_w2c=pose_init, device=torch.device("cuda")
        )
        pose_net.load_state_dict(nerf_ckpt['pose_net'], strict=True)

        nerf = Graph(settings, torch.device("cuda"), pose_net)
        nerf.load_state_dict(nerf_ckpt['nerf_net'], strict=True)
        nerf = nerf.eval().cuda()

        # Load input data
        input_data_path = os.path.join(project_path, "input_data.pickle")
        with open(input_data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data_dict = to_cuda(data_dict)

        # Analyze depth estimation
        mono_metrics, nerf_metrics, val_percent = analysis(
            data_dict, nerf, settings, th_number=2, relative_count=0, dense=False)

        mono_metrics_all.append(mono_metrics)
        nerf_metrics_all.append(nerf_metrics)
        val_percent_all.append(val_percent)
        seqcnt += 1

    # Stack and process metrics
    mono_metrics_all = np.stack(mono_metrics_all, axis=0)
    nerf_metrics_all = np.stack(nerf_metrics_all, axis=0)
    val_percent_all = np.stack(val_percent_all, axis=0)

    mono_metrics_all = average_metric(mono_metrics_all)
    nerf_metrics_all = average_metric(nerf_metrics_all)
    val_percent_all = np.mean(val_percent_all, axis=0)

    # Find best threshold
    th_dists = np.exp(np.linspace(np.log(0.05), np.log(0.0005), 200))
    th_param_to_report = [0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

    improvment_a10, cloest_idxs = [], []
    for th_param in th_param_to_report:
        cloest_idx = np.argmin(np.abs(th_dists - th_param))
        imprv = nerf_metrics_all[cloest_idx, 8] - mono_metrics_all[cloest_idx, 8]
        if np.isnan(imprv) or val_percent_all[cloest_idx] < 0.01:
            imprv = -np.inf
        improvment_a10.append(imprv)
        cloest_idxs.append(cloest_idx)

    best_imp_idx = np.argmax(improvment_a10)
    cloest_idx = cloest_idxs[best_imp_idx]

    # Format results
    error_depth_all = {
        'Mono': list(mono_metrics_all[cloest_idx]),
        'NeRF': list(nerf_metrics_all[cloest_idx])
    }

    table = [
        ['', 'sc_inv', 'log10', 'silog', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd05', 'd1', 'd2', 'd3', 'Density'],
    ]

    for key in error_depth_all.keys():
        table_entry = [key] + error_depth_all[key]
        table_entry.append(1.0 if key == 'Mono' else val_percent_all[cloest_idx])
        table.append(table_entry)

    print(f"=====Evaluated on {seqcnt} Seqs at Threshold {th_dists[cloest_idx]:.3f}====")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

if __name__ == '__main__':
    main()
