import glob
import os, sys, pickle
import argparse
import importlib

import kornia.geometry
import natsort
import numpy as np
import tqdm

proj_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, proj_root)

import cv2 as cv
import torch
import torch.backends.cudnn
from easydict import EasyDict as edict
from typing import Any, Dict
from tabulate import tabulate

import source.admin.settings as ws_settings
from analysis.utils_evaldepth import compute_depth_errors
from source.training.define_trainer import define_trainer
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.joint_pose_nerf_trainer import CommonPoseEvaluation
from source.training.core.triangulation_loss import padding_pose

def projection2corres(intr, poses, depthmap):
    nfrm, H, W = depthmap.shape
    gridxx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    gridyy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    gridxx = gridxx.view(1, H, W, 1)
    gridyy = gridyy.view(1, H, W, 1)
    gridpxls = torch.cat([gridxx, gridyy], dim=-1).float().cuda()

    intr, poses, depthmap = padding_pose(intr).cuda(), padding_pose(poses).cuda(), depthmap.cuda()
    selector = depthmap > 0
    corres = dict()
    for i in range(nfrm):
        for j in range(nfrm):
            if i == j:
                continue
            prjM = intr[j] @ poses[j] @ poses[i].inverse() @ intr[i].inverse()
            pts_source, depthf = gridpxls[0, selector[i], :], depthmap[i][selector[i]]
            pts_source_3D = kornia.geometry.convert_points_to_homogeneous(kornia.geometry.convert_points_to_homogeneous(pts_source) * depthf.unsqueeze(1))
            pts_target_2D = pts_source_3D @ prjM.transpose(-1, -2)
            pts_target_2D = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts_target_2D))
            corres[str([i, j])] = pts_target_2D
    return corres

def compute_depth_errors_batch(depth_gt, depth_est):
    metrics = list()
    for i in range(len(depth_gt)):
        metrics.append(compute_depth_errors(depth_gt[i].flatten(), depth_est[i].flatten()))
    metrics = np.stack(metrics, axis=0)
    return metrics

def evaluate_depthmap_scale_improvement(pose_net, data_dict):
    depth_est = data_dict['depth_est']
    depth_optimized = pose_net.optimize_depth(depth_est)
    depth_gt = data_dict['depth_gt']

    depth_metrics_est = compute_depth_errors_batch(depth_gt[1::], depth_est[1::])
    depth_metrics_opt = compute_depth_errors_batch(depth_gt[1::], depth_optimized[1::])

    depth_upperbound = list()
    for i in range(1, len(depth_gt)):
        selector = depth_gt[i] > 0
        gt_scale = np.median(depth_gt[i][selector]) / np.median(depth_est[i][selector])
        depth_upperbound.append(depth_est[i] * gt_scale)
    depth_upperbound = np.stack(depth_upperbound, axis=0)
    depth_metrics_upperbound = compute_depth_errors_batch(depth_gt[1::], depth_upperbound)
    return depth_metrics_est[:, 0:9], depth_metrics_opt[:, 0:9], depth_metrics_upperbound[:, 0:9]

def init_settings(train_module: str, train_name: str, args: Dict[str, Any]=None):

    settings = ws_settings.Settings(data_root='')
    train_module_for_launching = train_module

    # update with arguments
    # this is not very robust, assumes that it will be module/dataset, and want to add something here
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

    # does not support multi-gpu for now
    settings = edict(settings.__dict__)

    expr_module = importlib.import_module(
        'train_settings.{}.{}'.format(train_module_for_launching.replace('/', '.'), train_name.replace('/', '.'))
    )
    expr_func = getattr(expr_module, 'get_config')

    # get the config and define the trainer
    model_config = expr_func()

    settings.update(model_config)
    settings.train_sub = args.train_sub
    return settings


def pose_auc(errors, thresholds):
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

def evaluate_camera_relative(opt, pose_w2c, pose_GT_w2c):
    def comput_relpose(poses):
        from source.utils.camera import pad_poses, unpad_poses
        relposes = list()
        for i in range(len(poses)):
            relpose = unpad_poses(pad_poses(poses[0]) @ pad_poses(poses[i]).inverse())
            relpose[:, 3] = relpose[:, 3] / (relpose[:, 3] ** 2 + 1e-10).sum().sqrt()
            relposes.append(relpose)
        relposes = torch.stack(relposes, dim=0)
        return relposes

    relpose = comput_relpose(pose_w2c)
    relpose_GT = comput_relpose(pose_GT_w2c)

    def compute_relpose_error_deg(T_0to1_gt, T_0to1_est):
        # Compute Relative Pose Error
        def angle_error_mat(R1, R2):
            cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
            cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
            return np.rad2deg(np.abs(np.arccos(cos))).item()

        def angle_error_vec(v1, v2):
            n = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

        R_gt, t_gt = T_0to1_gt[:3, :3].cpu().numpy(), T_0to1_gt[:3, 3].cpu().numpy()
        R_est, t_est = T_0to1_est[:3, :3].cpu().numpy(), T_0to1_est[:3, 3].cpu().numpy()
        error_t = angle_error_vec(t_est.squeeze(), t_gt)
        error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation

        if np.isnan(error_t) or np.isinf(error_t):
            error_t = 180.0

        error_R = angle_error_mat(R_est, R_gt)
        return error_t, error_R

    error_pose = list()
    for i in range(1, len(pose_w2c)):
        error_t, error_R = compute_relpose_error_deg(relpose_GT[i], relpose[i])
        error_pose.append(max(error_t, error_R))
    error_pose = np.array(error_pose)

    # Align Pose
    evaluator = CommonPoseEvaluation()
    evaluator.settings = opt
    pose_aligned, _ = evaluator.prealign_w2c_small_camera_systems(opt, pose_w2c, pose_GT_w2c)
    error = evaluator.evaluate_camera_alignment(opt, pose_aligned, pose_GT_w2c)
    error_t = error['t'].cpu().numpy()
    return error_pose, error_t

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--data_root', type=str, default='/home/jupyter/shared',
                        help='Name of the train settings file.')
    parser.add_argument('--train_sub', type=int, default=None,
                        help='train subset: how many input views to consider?')

    parser.add_argument('--eval_entry', type=str, default=None,
                        help='which modality to evaluate?')
    args = parser.parse_args()

    split_export_path = os.path.join(proj_root, 'split', 'scannet', 'scannet.txt')
    with open(split_export_path) as file:
        entries = file.readlines()

    error_init_pose_rel_degs, error_opt_pose_rel_degs = list(), dict()
    error_init_pose_scale, error_opt_pose_scale = list(), dict()
    error_org_depth, error_opt_depth, error_upperbound_depth = list(), dict(), list()
    error_pose_all, error_depth_all = dict(), dict()

    # Evaluate Pose
    for entry in tqdm.tqdm(entries):
        seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
        args.scene = seq
        settings = init_settings(args.train_module, args.train_name, args)
        train_module_for_launching, train_module = args.train_module, args.train_module

        # update with arguments
        # this is not very robust, assumes that it will be module/dataset, and want to add something here
        base_dir_train_module = train_module.split('/')
        if args.train_sub is not None and args.train_sub != 0:
            base_dir_train_module[1] += '/subset_' + str(args.train_sub)
        else:
            args.train_sub = None

        if args.scene is not None:
            base_dir_train_module[1] += '/' + args.scene
        train_module = '/'.join(base_dir_train_module)

        project_path = os.path.join(proj_root, 'checkpoint', train_module, args.train_name)

        # init_pose_colmnap_path = os.path.join(project_path.replace(args.train_name, 'batch_local_sfm_colmap'), "init_pose.pickle")
        # with open(init_pose_colmnap_path, 'rb') as f:
        #     init_poses_w2c_colmap = pickle.load(f)
        #     if torch.sum(torch.isnan(init_poses_w2c_colmap)) > 0:
        #         continue

        if seq == "scene0794_00":
            # nan in gt pose
            continue

        # Read input Data
        data_path = os.path.join(project_path, "input_data.pickle")
        with open(data_path, 'rb') as f:
            input_data = pickle.load(f)

        # Evaluate Init Pose Estimation
        init_pose_path = os.path.join(project_path, "init_pose.pickle")
        with open(init_pose_path, 'rb') as f:
            init_poses_w2c = pickle.load(f)
        gt_poses_w2c = input_data['pose']
        error_init_pose, error_init_pose_t = evaluate_camera_relative(settings, init_poses_w2c, gt_poses_w2c)
        error_init_pose_rel_degs.append(error_init_pose), error_init_pose_scale.append(error_init_pose_t)

        # Evaluate Rest Pose Estimation
        pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            settings, nbr_poses=args.train_sub, initial_poses_w2c=init_poses_w2c, device=torch.device("cpu")
        )
        opted_poses = glob.glob(os.path.join(project_path, "*.pth"))
        opted_poses_ = []
        for x in opted_poses:
            if 'pose_optimized' in x:
                opted_poses_.append(x)
        opted_poses = opted_poses_

        opted_poses = natsort.natsorted(opted_poses)
        for opted_pose in opted_poses:
            opted_pose_dict = torch.load(opted_pose)
            pose_net.load_state_dict(opted_pose_dict, strict=True)
            opted_pose_estimate = pose_net.get_w2c_poses()
            error_opted_pose_estimate, error_opted_pose_estimate_t = evaluate_camera_relative(settings, opted_pose_estimate, gt_poses_w2c)

            opted_pose_key = opted_pose.split('/')[-1].split('.')[0]
            if opted_pose_key not in error_opt_pose_rel_degs:
                error_opt_pose_rel_degs[opted_pose_key] = list()
                error_opt_pose_scale[opted_pose_key] = list()
                error_opt_depth[opted_pose_key] = list()
            error_opt_pose_rel_degs[opted_pose_key].append(error_opted_pose_estimate)
            error_opt_pose_scale[opted_pose_key].append(error_opted_pose_estimate_t)

            # Evaluate Depth
            depth_metrics_est, depth_metrics_opt, depth_metrics_upperbound = evaluate_depthmap_scale_improvement(pose_net, input_data)
            error_opt_depth[opted_pose_key].append(depth_metrics_opt)

            # # Evaluate Correspondence Under Optimized Pose
            # intr, w2c_poses_gt, w2c_poses_et, depthmap = input_data['intr'], input_data['pose'], opted_pose_estimate, input_data['depth_gt']
            # corres_prj_gt = projection2corres(intr, w2c_poses_gt, depthmap)
            # corres_prj_et = projection2corres(intr, w2c_poses_et, depthmap)
            # corres_mah_et = pose_net

        if len(opted_poses) > 0:
            error_org_depth.append(depth_metrics_est)
            error_upperbound_depth.append(depth_metrics_upperbound)

    error_init_pose_scale = np.concatenate(error_init_pose_scale)
    print("nan value in init as %f" % (np.sum(np.isnan(error_init_pose_scale))))

    error_pose_all['init_pose'] = pose_auc(np.concatenate(error_init_pose_rel_degs), [5, 10, 20])
    error_pose_all['init_pose'].append(error_init_pose_scale.mean())

    for x in error_opt_pose_rel_degs.keys():
        print("Name %s has %d estimates" % (x, len(error_opt_pose_rel_degs[x])))
        error_pose_all[x] = pose_auc(np.concatenate(error_opt_pose_rel_degs[x]), [5, 10, 20])
        error_opt_pose_scale_x = np.concatenate(error_opt_pose_scale[x])
        error_pose_all[x].append(error_opt_pose_scale_x.mean())

    print('%s Init Pose Performance' % (args.train_name))
    table = [
        ['', 'Auc-5', 'Auc-10', 'Auc-20', 'Error_t'],
    ]
    for key in error_pose_all.keys():
        table_entry = [key] + error_pose_all[key]
        table.append(table_entry)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

    if len(opted_poses) > 0:
        error_org_depth = np.concatenate(error_org_depth, axis=0)
        error_depth_all['org_mono'] = list(np.mean(error_org_depth, axis=0))
        error_upperbound_depth = np.concatenate(error_upperbound_depth, axis=0)
        error_depth_all['uppperbound_mono'] = list(np.mean(error_upperbound_depth, axis=0))

        for x in error_opt_depth.keys():
            error_depth_all[x] = np.concatenate(error_opt_depth[x], axis=0)
            error_depth_all[x] = list(np.mean(error_depth_all[x], axis=0))
        table = [
            ['', 'sc_inv', 'log10', 'silog', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd05', 'd1'],
        ]
        for key in error_depth_all.keys():
            table_entry = [key] + error_depth_all[key]
            table.append(table_entry)
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))


if __name__ == '__main__':
    main()
