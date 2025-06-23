import einops, kornia
import copy, time, secrets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import source.utils.camera as camera
from typing import Dict, Any, Optional
from third_party.LightedDepth.GPUEPMatrixEstimation import gpuepm_function_topk
from source.training.core.triangulation_loss import padding_pose
from kornia.geometry.epipolar.essential import cross_product_matrix

from source.training.core.bawotriang import epp_scoring_function, prj_scoring, c3D_scoring


@torch.no_grad()
def evaluate_camera_relative(pose_w2c, pose_GT_w2c):
    def comput_relpose(poses):
        from source.utils.camera import pad_poses, unpad_poses
        relposes = list()
        for i in range(len(poses)):
            relpose = unpad_poses(pad_poses(poses[0]) @ pad_poses(poses[i]).inverse())
            relpose[:, 3] = relpose[:, 3] / (relpose[:, 3] ** 2 + 1e-10).sum().sqrt()
            relposes.append(relpose)
        relposes = torch.stack(relposes, dim=0)
        return relposes

    relpose = pose_w2c[0:1] @ pose_w2c.inverse()
    relpose_GT = padding_pose(comput_relpose(pose_GT_w2c))

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

    errors = list()
    for i in range(1, len(pose_w2c)):
        errors_i = list()
        for j in range(relpose[i].shape[0]):
            error_t, error_R = compute_relpose_error_deg(relpose_GT[i], relpose[i, j])
            error = max(error_t, error_R)
            errors_i.append(error)
        errors_i = np.array(errors_i)
        errors.append(errors_i)
    errors = np.stack(errors, axis=0)

    return errors

def pose_to_d10(pose: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrix to 9D representation.

    We take the two first ROWS of the rotation matrix,
    along with the translation vector.
    ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
    """
    nbatch = pose.shape[0]
    R = pose[:, :3, :3]  # [N, 3, 3]
    t = pose[:, :3, -1]  # [N, 3]
    scale = (torch.sum(t ** 2, dim=1) + 1e-16).sqrt()
    scale = scale.unsqueeze(1)
    t = t / scale

    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    d10 = torch.cat((t, r6, scale), -1)  # [N, 9]
    # first is the translation vector, then two first ROWS of rotation matrix

    return d10


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
            first two rows of the rotation matrix.
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row

def divide_into_R_nt_scale(pose_candidates, eps):
    R, t = torch.split(pose_candidates[..., 0:3, :], [3, 1], dim=-1)
    scale = torch.sqrt(torch.sum(t ** 2, dim=-2, keepdim=True) + eps)
    nt = t / scale
    zeroscale = scale < 1e-3
    scale[zeroscale] = 0
    nt = torch.zeros_like(nt) + nt * (1 - zeroscale.float())
    return R, nt, scale

def ablate_scale(posei_candidates, posej_candidates, eps, nablate_epp):
    posei_candidates, posej_candidates = posei_candidates.double(), posej_candidates.double()
    npair, nenum, _, _ = posei_candidates.shape
    posei_candidates, posej_candidates = posei_candidates.view([npair, nenum, 4, 4]), posej_candidates.view([npair, nenum, 4, 4])
    Ri, nti, scalei = divide_into_R_nt_scale(posei_candidates, eps)
    Rj, ntj, scalej = divide_into_R_nt_scale(posej_candidates, eps)

    Rc = Rj @ Ri.transpose(-1, -2)
    ntc1, ntc2 = ntj, -Rc @ nti
    ntc2 = ntc2 / torch.sqrt(torch.sum(ntc2 ** 2, dim=-2, keepdim=True) + eps)

    theta = torch.arccos(torch.sum(ntc1 * ntc2, dim=-2, keepdim=True))
    ablated_w = torch.linspace(0, 1, nablate_epp).to(posei_candidates.device)
    ablate_theta = theta.view([npair, nenum, 1, 1, 1]) * ablated_w.view([1, 1, nablate_epp, 1, 1])

    x_axis_val, y_axis_val = torch.cos(ablate_theta), torch.sin(ablate_theta)
    vec_perpen_ntc1_ntc2 = torch.cross(ntc1, ntc2, dim=-2)
    x_axis_vec, y_axis_vec = ntc1, torch.cross(vec_perpen_ntc1_ntc2, ntc1, dim=-2)
    y_axis_vec = y_axis_vec / torch.sqrt(torch.sum(y_axis_vec ** 2, dim=-2, keepdim=True) + eps)

    x_axis_vec, y_axis_vec = x_axis_vec.view([npair, nenum, 1, 3, 1]), y_axis_vec.view([npair, nenum, 1, 3, 1])
    ntc = x_axis_vec * x_axis_val + y_axis_vec * y_axis_val
    ntc = ntc / torch.sqrt(torch.sum(ntc ** 2, dim=-2, keepdim=True) + eps)

    ntc_degenerate = (ntj - Rc @ nti).view(  [npair, nenum, 1, 3, 1])
    scalei_zero = (scalei < eps).float().view([npair, nenum, 1, 1, 1])
    scalej_zero = (scalej < eps).float().view([npair, nenum, 1, 1, 1])
    # assert torch.sum(scalei_zero * scalej_zero) == 0
    scale_nonzero = (scalei_zero == 0) * (scalej_zero == 0)
    scale_nonzero = scale_nonzero.float()
    ntc = ntc * scale_nonzero + ntc_degenerate * (1 - scale_nonzero)

    # # Check for Correctness
    # ntc1idx, ntc2idx = 0, nablate_epp-1
    # res1 = (ntc[..., ntc1idx, :, :] - ntc1).abs()
    # res1 = (res1 * scale_nonzero.view([npair, nenum, 1, 1])).max()
    # res2 = (ntc[..., ntc2idx, :, :] - ntc2).abs()
    # res2 = (res2 * scale_nonzero.view([npair, nenum, 1, 1])).max()
    # assert res1 < 1e-3 and res2 < 1e-3

    Rc, ntc = Rc.view([npair, nenum, 1, 3, 3]), ntc.view([npair, nenum, nablate_epp, 3, 1])
    theta = theta.view([npair, nenum, 1, 1, 1])
    return Rc.float(), ntc.float(), theta.float()


class FirstTwoColunmnsScalePoseOptDepthParameters(nn.Module):
    def __init__(self, opt: Dict[str, Any], nbr_poses: int, initial_poses_w2c: torch.Tensor,
                 device: torch.device):
        super().__init__()

        self.opt = opt
        self.optimize_c2w = self.opt.camera.optimize_c2w
        self.optimize_trans = self.opt.camera.optimize_trans
        self.optimize_rot = self.opt.camera.optimize_rot

        self.nbr_poses = nbr_poses  # corresponds to the total number of poses!
        self.device = device

        self.initial_poses_w2c = initial_poses_w2c  # corresponds to initialization of all the poses
        # including the ones that are fixed
        self.initial_poses_c2w = camera.pose.invert(self.initial_poses_w2c)

        # [N, 9]: (x, y, z, r1, r2) or [N, 3]: (x, y, z)
        self.init_poses_embed()
        # here pose_embedding is camera to world!!
        self.init_depth_embed()

        if not hasattr(self.opt, 'nablate_epp'):
            return

        self.eps, self.topk, self.nablate_epp, self.nablate_prj = 1e-10, 10, self.opt.nablate_epp, self.opt.nablate_prj
        self.topk, self.max_res_scale, self.prj_th, self.c3D_th = self.opt.topk, self.opt.max_res_scale, float(self.opt.prj_th), float(self.opt.c3D_th)

        self.document_compare_embedding(initialization=True)
        self.max_count = 0

        self.skip_epp_votes, self.skip_prj_votes, self.skip_c3D_votes = self.opt.skip_epp_votes, self.opt.skip_prj_votes, self.opt.skip_c3D_votes
        if hasattr(self.opt, 'c3D_w'):
            self.c3D_w = self.opt.c3D_w
        else:
            self.c3D_w = 0.0
        self.c3D_w = 1.0

        if hasattr(self.opt, 'nptsw'):
            self.nptsw = self.opt.nptsw
        else:
            self.nptsw = False

    def init_depth_embed(self):
        depth_embedding_rest = torch.ones([self.opt.train_sub - 1]).to(self.device)
        self.depth_embedding_rest = nn.Parameter(depth_embedding_rest, requires_grad=False)
        depth_embedding_root = torch.ones([1]).to(self.device)
        self.depth_embedding_root = nn.Parameter(depth_embedding_root, requires_grad=False)

    def init_poses_embed(self):
        poses_w2c_to_optimize = self.initial_poses_w2c
        poses_w2c_to_optimize_embed = pose_to_d10(poses_w2c_to_optimize)
        self.pose_embedding = nn.Parameter(poses_w2c_to_optimize_embed, requires_grad=False)

    def get_initial_w2c(self):
        return self.initial_poses_w2c

    def get_c2w_poses(self, detach_scale=False, detach_npose=False) -> torch.Tensor:
        w2c = self.get_w2c_poses(detach_scale=detach_scale, detach_npose=detach_npose)
        poses_c2w = camera.pose.invert(w2c)
        return poses_c2w

    def get_w2c_poses(self, detach_scale=False, detach_npose=False) -> torch.Tensor:
        if self.optimize_rot and self.optimize_trans:
            t = self.pose_embedding[:, :3]  # [n_to_optimize, 3]
            r = self.pose_embedding[:, 3:9]  # [n_to_optimize, 6]
            scale = self.pose_embedding[:, 9:10]
            t = t / torch.sqrt(torch.sum(t ** 2, dim=1, keepdim=True) + 1e-10)
            if detach_scale:
                scale = scale.detach()
            if detach_npose:
                t, r = t.detach(), r.detach()
        R = r6d2mat(r)[:, :3, :3]  # [n_to_optimize, 3, 3]
        poses_w2c = torch.cat((R, t[..., None] * scale[..., None]), -1)  # [n_to_optimize, 3, 4]
        return poses_w2c

    @torch.no_grad()
    def register_inputs(
            self, corres_estimate_bundle, train_data, norm_threshold_adjuster=0.4, visibility_threshold=0.1
    ):

        intr = train_data['intr'][0]
        self.intr = intr
        self.norm_threshold, self.visibility_threshold = norm_threshold_adjuster / intr[:2, :2].abs().mean().item(), visibility_threshold

        # Register the Correspondence between two Cam Views
        flatten_npose_initer, flatten_epp_evaluater, flatten_prj_evaluator = dict(), dict(), dict()
        corres_maps, flow_pairs, conf_maps, mask_valid_corr = corres_estimate_bundle
        depth_est = train_data['depth_est']

        self.flow_pairs_to_idx = dict()
        for idx, pair in enumerate(flow_pairs):
            self.flow_pairs_to_idx[str(pair)] = idx

        N, _, H, W = corres_maps.shape
        gridxx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        gridyy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        gridxx = gridxx.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()
        gridyy = gridyy.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()
        gridpxls = torch.cat([gridxx, gridyy], dim=1)
        gridpxls = gridpxls.expand([N, -1, -1, -1])

        # Compute Visibility
        index_i, index_j = list(), list()
        for pairidx, (i, j) in enumerate(flow_pairs):
            index_i.append(i), index_j.append(j)
        index_i, index_j = torch.from_numpy(np.array(index_i)), torch.from_numpy(np.array(index_j))
        depth_viewi, depth_viewj = depth_est[index_i], depth_est[index_j]
        visibility_maps = (corres_maps[:, 0:1] > 0) * (corres_maps[:, 0:1] < W - 1) * (corres_maps[:, 1:2] > 0) * (corres_maps[:, 1:2] < H - 1) * \
                          (depth_viewi.unsqueeze(1) > 0) * (conf_maps > visibility_threshold)

        depth_viewj_valid = (depth_viewj > 0).float()
        corrx, corry = torch.split(corres_maps, 1, dim=1)
        corrx, corry = (corrx / (W - 1) - 0.5) * 2, (corry / (H - 1) - 0.5) * 2
        splx_coord = torch.stack([corrx, corry], dim=-1).view([len(index_i), H, W, 2])
        depth_viewj_aligned = torch.nn.functional.grid_sample(depth_viewj.unsqueeze(1), splx_coord, align_corners=True, mode='bilinear')
        depth_viewj_aligned = depth_viewj_aligned.view([len(index_i), H, W])
        depth_viewj_aligned_valid = torch.nn.functional.grid_sample(depth_viewj_valid.unsqueeze(1), splx_coord, align_corners=True, mode='bilinear')
        depth_viewj_aligned_valid = depth_viewj_aligned_valid > 0.99
        visibility_maps = visibility_maps * depth_viewj_aligned_valid

        def flatten_estimate(
                corres_source, corres_target, conf_maps, mask_valid_corr, pairidx, depth_viewi, depth_viewj
        ):
            cor_source, cor_target, confidence, mask = corres_source[pairidx], corres_target[pairidx], conf_maps[pairidx], mask_valid_corr[pairidx]
            cor_source, cor_target, confidence, mask = cor_source.unsqueeze(0), cor_target.unsqueeze(0), confidence.unsqueeze(0), mask.unsqueeze(0)
            cor_sourcef = einops.rearrange(cor_source, 'b c h w -> (b h w) c')
            cor_targetf = einops.rearrange(cor_target, 'b c h w -> (b h w) c')
            confidencef = einops.rearrange(confidence, 'b c h w -> (b h w) c')
            selector = einops.rearrange(mask, 'b c h w -> (b c h w)')

            cor_sourcef, cor_targetf, confidencef = cor_sourcef[selector], cor_targetf[selector], confidencef[selector]

            depth_viewif = einops.rearrange(depth_viewi, 'h w -> (h w)')[selector]
            depth_viewjf = einops.rearrange(depth_viewj, 'h w -> (h w)')[selector]
            return torch.cat([cor_sourcef, cor_targetf], dim=1), confidencef.squeeze(), depth_viewif, depth_viewjf

        if self.nptsw:
            self.visibility_w = torch.sum(visibility_maps, dim=[1, 2, 3]) / torch.sum(visibility_maps)
            self.visibility_w = self.visibility_w.view([len(visibility_maps), 1])
        else:
            self.visibility_w = torch.ones([len(visibility_maps)], device=visibility_maps.device)
            self.visibility_w = self.visibility_w.view([len(visibility_maps), 1])

        for pairidx, (i, j) in enumerate(flow_pairs):
            # Evaluate Epipolar Geometry Scoring Function
            corres_itoj, confidencef_itoj, _, _ = flatten_estimate(
                gridpxls, corres_maps, conf_maps, mask_valid_corr, pairidx, depth_viewi[pairidx], depth_viewj_aligned[pairidx]
            )

            assert str([i, j]) not in flatten_npose_initer
            flatten_npose_initer[str([i, j])] = [corres_itoj, confidencef_itoj]

            assert str([i, j]) not in flatten_epp_evaluater
            flatten_epp_evaluater[str([i, j])] = self.random_sampling_wthconf(corres_itoj, confidencef_itoj, forval=True)

            # Evaluate Projection Geometry Scoring Function
            corres_itoj, _, depthmap_i, depthmap_j = flatten_estimate(
                gridpxls, corres_maps, conf_maps, visibility_maps, pairidx, depth_viewi[pairidx], depth_viewj_aligned[pairidx]
            )

            assert str([i, j]) not in flatten_prj_evaluator
            flatten_prj_evaluator[str([i, j])] = self.random_sampling_woconf(corres_itoj, depthmap_i, depthmap_j)


        self.flatten_npose_initer, self.flatten_epp_evaluater, self.flatten_prj_evaluator = flatten_npose_initer, flatten_epp_evaluater, flatten_prj_evaluator
        self.flow_pairs_to_idx, self.flow_pairs = dict(), flow_pairs
        for idx, pair in enumerate(flow_pairs):
            self.flow_pairs_to_idx[str(pair)] = idx

        self.depth_est, self.corres_estimate_bundle = depth_est, corres_estimate_bundle

        # Precompute Relative Pose Searching Pool
        nfrm = len(self.depth_est)
        self.relpose_itoj_candidates_rec = dict()
        for frmi in range(nfrm):
            for frmj in range(nfrm):
                if frmj != frmi:
                    # Sample from the two frames
                    frmi_frmj_corres, frmi_conf = self.flatten_npose_initer[str([frmi, frmj])]
                    frmi_frmj_corres = self.random_sampling_wthconf(frmi_frmj_corres, frmi_conf, forval=True)
                    frmj_frmi_corres, frmj_conf = self.flatten_npose_initer[str([frmj, frmi])]
                    frmj_frmi_corres = self.random_sampling_wthconf(frmj_frmi_corres, frmj_conf, forval=True)

                    # Init the relative pose candidates
                    frmi_coor_fromij, frmj_coor_fromij = torch.split(frmi_frmj_corres, 2, dim=1)
                    frmj_coor_fromji, frmi_coor_fromji = torch.split(frmj_frmi_corres, 2, dim=1)
                    coor_source = torch.cat([frmi_coor_fromij, frmi_coor_fromji], dim=0)
                    coor_target = torch.cat([frmj_coor_fromij, frmj_coor_fromji], dim=0)

                    coor_source = kornia.geometry.conversions.convert_points_to_homogeneous(coor_source)
                    coor_target = kornia.geometry.conversions.convert_points_to_homogeneous(coor_target)
                    coor_source, coor_target = (intr.inverse() @ coor_source.T).T, (intr.inverse() @ coor_target.T).T
                    coor_source = kornia.geometry.conversions.convert_points_from_homogeneous(coor_source)
                    coor_target = kornia.geometry.conversions.convert_points_from_homogeneous(coor_target)

                    relpose_itoj_candidates = gpuepm_function_topk(
                        coor_source, coor_target, ransac_iter=5, ransac_threshold=self.norm_threshold, topk=self.topk
                    )
                    self.relpose_itoj_candidates_rec[str([frmi, frmj])] = padding_pose(relpose_itoj_candidates)
                else:
                    self.relpose_itoj_candidates_rec[str([frmi, frmj])] = torch.eye(4).view([1, 4, 4]).repeat([self.topk, 1, 1]).cuda()

    def random_sampling_wthconf(self, correspondencef, confidencef, sample_num=10000, forval=False):
        # Set replace True to prevent insufficient points
        confidencef[confidencef < self.opt.min_conf_valid_corr] = 0.0
        if forval:
            # Ensure same as Evaluation
            np.random.seed(int(torch.sum(confidencef).int().item() * 100))

        confidencef = confidencef.cpu().numpy()
        good_samples = np.random.choice(
            np.arange(len(correspondencef)),
            size=sample_num,
            replace=True,
            p=confidencef / np.sum(confidencef),
        )
        correspondencef_ = correspondencef[good_samples, :]

        return correspondencef_

    def random_sampling_woconf(self, correspondencef, depthmap_if, depthmap_jf, sample_num=10000):
        good_samples = np.random.choice(
            np.arange(len(depthmap_if)),
            size=sample_num,
            replace=True,
        )
        correspondencef, depthmap_if, depthmap_jf = correspondencef[good_samples, :], depthmap_if[good_samples], depthmap_jf[good_samples]
        return correspondencef, depthmap_if, depthmap_jf

    @torch.no_grad()
    def ransac_update_on_frame_i(self, renderer, frmi, nfrm, data_dict, seed, foreval=False, noreroot=False):

        """
        np.random.seed(int(time.time()) + seed)
        while True:
            frmj = np.random.randint(nfrm)
            if frmj != frmi:
                break

        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
        from analysis.utils_vls import tensor2rgb
        image = data_dict['image']
        fig1, fig2 = tensor2rgb(image, viewind=frmi), tensor2rgb(image, viewind=frmj)

        corres1, _ = self.flatten_npose_initer[str([frmi, frmj])]
        rndidx1 = np.random.randint(0, len(corres1))
        corres2 = self.flatten_epp_evaluater[str([frmi, frmj])]
        rndidx2 = np.random.randint(0, len(corres2))
        corres3, depth3, depth3r = self.flatten_prj_evaluator[str([frmi, frmj])]
        rndidx3 = np.random.randint(0, len(depth3))
        intr, pose_i, pose_j = padding_pose(data_dict['intr'])[0], padding_pose(self.get_w2c_poses())[frmi], padding_pose(self.get_w2c_poses())[frmj]
        prjM = intr @ pose_j @ pose_i.inverse() @ intr.inverse()
        pts3d = kornia.geometry.convert_points_to_homogeneous(kornia.geometry.convert_points_to_homogeneous(corres3[rndidx3:rndidx3+1, 0:2]) * depth3[rndidx3])
        pts2d = pts3d @ prjM.T
        pts2d = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts2d))
        prjMr = prjM.inverse()
        pts3dr = kornia.geometry.convert_points_to_homogeneous(kornia.geometry.convert_points_to_homogeneous(corres3[rndidx3:rndidx3 + 1, 2:4]) * depth3r[rndidx3])
        pts2dr = pts3dr @ prjMr.T
        pts2dr = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts2dr))

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs[0].imshow(fig1)
        axs[0].scatter(corres1[rndidx1, 0].item(), corres1[rndidx1, 1].item())
        axs[0].scatter(corres2[rndidx2, 0].item(), corres2[rndidx2, 1].item())
        axs[0].scatter(corres3[rndidx3, 0].item(), corres3[rndidx3, 1].item())
        axs[0].scatter(pts2dr[0, 0].item(), pts2dr[0, 1].item())
        axs[1].imshow(fig2)
        axs[1].scatter(corres1[rndidx1, 2].item(), corres1[rndidx1, 3].item())
        axs[1].scatter(corres2[rndidx2, 2].item(), corres2[rndidx2, 3].item())
        axs[1].scatter(pts2d[0, 0].item(), pts2d[0, 1].item())
        plt.show()
        """

        # Set Optimization Root
        if noreroot:
            optroot = frmi
        else:
            optroot = 0
        current_root = frmi

        # Current Estimate of Pose rooted by frmi
        w2c_poses = padding_pose(self.get_w2c_poses())
        w2c_poses_gt = padding_pose(data_dict['pose'])
        intr = data_dict['intr'][0]

        # Move to Root on frmi
        if noreroot:
            w2c_poses = w2c_poses @ w2c_poses[frmi:frmi+1].inverse()

        scale_itorest, scale_itorest_gt = dict(), dict()
        for frmj in range(nfrm):
            if frmj != frmi:
                pose_itoj = w2c_poses[frmj] @ w2c_poses[frmi].inverse()
                scale_itorest[str([frmi, frmj])] = torch.sqrt(torch.sum(pose_itoj[0:3, 3] ** 2) + self.eps).item()
                pose_itoj_gt = w2c_poses_gt[frmj] @ w2c_poses_gt[frmi].inverse()
                scale_itorest_gt[str([frmi, frmj])] = torch.sqrt(torch.sum(pose_itoj_gt[0:3, 3] ** 2) + self.eps).item()

        pose_itoj_candidates_rec = dict()
        for frmj in range(nfrm):
            if frmj != frmi:
                pose_itoj_candidates = copy.deepcopy(self.relpose_itoj_candidates_rec[str([frmi, frmj])])[0:self.topk]
                pose_itoj_candidates[:, 0:3, 3] = pose_itoj_candidates[:, 0:3, 3] * scale_itorest[str([frmi, frmj])]
                pose_itoj_candidates_rec[str([frmi, frmj])] = pose_itoj_candidates
            else:
                pose_itoj_candidates_rec[str([frmi, frmj])] = copy.deepcopy(self.relpose_itoj_candidates_rec[str([frmi, frmj])])[0:self.topk]

        # Organize as root frame is index 0
        pose_all_candidates, rootidx = list(), 0
        for frmj in range(nfrm):
            if noreroot:
                pose_itoj_candidates = pose_itoj_candidates_rec[str([frmi, frmj])]
                pose_itoj_current = w2c_poses[frmj].view([1, 4, 4])
                pose_all_candidates.append(torch.cat([pose_itoj_current, pose_itoj_candidates], dim=0))
            else:
                pose_itoj_candidates = pose_itoj_candidates_rec[str([frmi, frmj])]
                pose_roottoi_current = w2c_poses[frmj].view([1, 4, 4])
                pose_roottoi_candidates = pose_itoj_candidates @ w2c_poses[frmi]
                pose_all_candidates.append(torch.cat([pose_roottoi_current, pose_roottoi_candidates], dim=0))

        pose_all_candidates = torch.stack(pose_all_candidates, dim=0)

        # Organize into different Batch
        batchify_poses = list()
        for i in range(1, nfrm):
            pose_ransac = pose_all_candidates[i].view([1, self.topk+1, 4, 4])
            pose_stored = w2c_poses.view([nfrm, 1, 4, 4])
            selector = (torch.arange(0, nfrm) == i).float().cuda().view([nfrm, 1, 1, 1])
            perbatch_poses = pose_ransac * selector + pose_stored * (1 - selector)
            perbatch_poses = torch.cat([w2c_poses.view([nfrm, 1, 4, 4]), perbatch_poses], dim=1)
            batchify_poses.append(perbatch_poses)
        batchify_poses = torch.cat(batchify_poses, dim=1)
        pose_all_candidates = batchify_poses

        # rnd0 = np.random.randint(1, nfrm)
        # enumerated_poses = batchify_poses[rnd0, 1+(rnd0-1)*(self.topk+1):1+(rnd0-1+1)*(self.topk+1)]
        # enumerated_poses_diff = torch.max(torch.abs(enumerated_poses - pose_all_candidates[rnd0]))

        if foreval:
            error_all = evaluate_camera_relative(pose_all_candidates, w2c_poses_gt)
            return error_all

        # Produce Correspondence and Depth for Scoring Function Computation
        epp_corres, prj_corres_depth = list(), list()
        for frmi in range(nfrm):
            for frmj in range(nfrm):
                if frmi != frmj:
                    corres, depth_viewi, depth_viewj = self.flatten_prj_evaluator[str([frmi, frmj])]
                    prj_corres_depth.append(torch.cat([corres, depth_viewi.unsqueeze(-1), depth_viewj.unsqueeze(-1)], dim=-1))
                    epp_corres.append(self.flatten_epp_evaluater[str([frmi, frmj])])
        epp_corres, prj_corres_depth = torch.stack(epp_corres), torch.stack(prj_corres_depth)

        """
        # Evaluate the Correctness of the Sampling
        while True:
            frmi, frmj = np.random.randint(nfrm), np.random.randint(nfrm)
            if frmj != frmi:
                break

        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
        from analysis.utils_vls import tensor2rgb
        image = data_dict['image']
        fig1, fig2 = tensor2rgb(image, viewind=frmi), tensor2rgb(image, viewind=frmj)

        index = self.flow_pairs_to_idx[str([frmi, frmj])]
        rndidx1 = np.random.randint(len(epp_corres[index]))
        corres1 = epp_corres[index]

        rndidx2 = np.random.randint(len(prj_corres_depth[index]))
        corres2_depth = prj_corres_depth[index, 0]
        intr, pose_i, pose_j = padding_pose(data_dict['intr'])[0], pose_all_candidates[frmi, 1], pose_all_candidates[frmj, 1]
        prjM = intr @ pose_j @ pose_i.inverse() @ intr.inverse()
        pts3d = kornia.geometry.convert_points_to_homogeneous(kornia.geometry.convert_points_to_homogeneous(
            corres2_depth[rndidx2:rndidx2+1, 0:2]) * corres2_depth[rndidx2, 4])
        pts2d = pts3d @ prjM.T
        pts2d = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts2d))

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs[0].imshow(fig1)
        axs[0].scatter(corres1[rndidx1, 0].item(), corres1[rndidx1, 1].item())
        axs[0].scatter(corres2_depth[rndidx2, 0].item(), corres2_depth[rndidx2, 1].item())
        axs[1].imshow(fig2)
        axs[1].scatter(corres1[rndidx1, 2].item(), corres1[rndidx1, 3].item())
        axs[1].scatter(pts2d[0, 0].item(), pts2d[0, 1].item())
        plt.show()
        """

        # Compute the relative camera
        index_frmi, index_frmj = list(), list()
        for frmi in range(nfrm):
            for frmj in range(nfrm):
                if frmi == frmj:
                    continue
                index_frmi.append(frmi)
                index_frmj.append(frmj)
        index_frmi, index_frmj = torch.Tensor(index_frmi).long(), torch.Tensor(index_frmj).long()
        posei_candidates = pose_all_candidates[index_frmi]
        posej_candidates = pose_all_candidates[index_frmj]

        # Ablate on Camera Scale, theta is used for indexing
        Rc, ntc, theta = ablate_scale(posei_candidates, posej_candidates, self.eps, self.nablate_epp)

        """
        # Verify Max Angle Ambiguity Due to Voting Resolution
        pose_ref_itoj = posej_candidates @ posei_candidates.inverse()
        ck_R_ref_itoj = pose_ref_itoj[..., None, 0:3, 0:3].transpose(-1, -2) @ Rc
        ck_R_ref_itoj = (ck_R_ref_itoj - torch.eye(3).cuda().view([1, 1, 1, 3, 3])).abs().max()
        ck_t_ref_itoj = pose_ref_itoj[..., 0:3, 3:4]
        ck_t_ref_itoj = ck_t_ref_itoj / torch.sqrt(torch.sum(ck_t_ref_itoj ** 2, dim=[2, 3], keepdim=True) + self.eps)
        ck_t_ref_itoj = (ck_t_ref_itoj.unsqueeze(-3) * ntc).sum(dim=-2)
        ck_t_ref_itoj = torch.clamp(ck_t_ref_itoj, min=-1+self.eps, max=1-self.eps)
        ck_t_ref_itoj = torch.rad2deg(torch.arccos(ck_t_ref_itoj))
        ck_t_ref_itoj, _ = torch.min(ck_t_ref_itoj, dim=-2)
        ck_t_ref_itoj = ck_t_ref_itoj.max()

        # Per Index Verification, Relative Pose: Frmi - indexi to Frmj - indexj
        npair, npts, _ = epp_corres.shape
        _, nenum, _, _ = posei_candidates.shape
        while True:
            frmi, frmj = np.random.randint(nfrm), np.random.randint(nfrm)
            if frmj != frmi:
                break
        pose_idx = np.random.randint(nenum)
        pose_ref_i, pose_ref_j = pose_all_candidates[frmi, pose_idx], pose_all_candidates[frmj, pose_idx]
        pose_ref_itoj = pose_ref_j @ pose_ref_i.inverse()

        index_ref = self.flow_pairs_to_idx[str([frmi, frmj])]
        Rc_ref, ntc_ref = Rc[index_ref, pose_idx, 0], ntc[index_ref, pose_idx, :, :].squeeze(-1)

        ck_R_ref = (Rc_ref @ pose_ref_itoj[0:3, 0:3].T - torch.eye(3).cuda()).abs().max()
        ck_t_ref = pose_ref_itoj[0:3, 3] / ((pose_ref_itoj[0:3, 3] ** 2) + self.eps).sum().sqrt()
        ck_t_ref = torch.rad2deg(torch.arccos(torch.sum(ck_t_ref.view([1, 3]) * ntc_ref, dim=-1))).min()
        """

        # CUDA Compute Epipolar Scoring Function
        # st = time.time()

        to_compute_pair = self.document_compare_embedding(compare=True, root=current_root)
        to_compute_pair = to_compute_pair.int()

        npair, npts, _ = epp_corres.shape
        nenum = ntc.shape[1]
        _, topk_plus1, _, _ = posei_candidates.shape
        intr = intr.view([1, 1, 3, 3])
        essen_mtx = cross_product_matrix(ntc.view([npair, nenum, self.nablate_epp, 3])) @ Rc
        pts_source, pts_target = torch.split(epp_corres, 2, dim=2)
        pts_source, pts_target = kornia.geometry.conversions.convert_points_to_homogeneous(pts_source), kornia.geometry.conversions.convert_points_to_homogeneous(pts_target)
        pts_source, pts_target = pts_source.view([npair, npts, 3, 1]), pts_target.view([npair, npts, 3, 1])
        pts_source_normed, pts_target_normed = intr.inverse() @ pts_source, intr.inverse() @ pts_target
        if self.skip_epp_votes:
            epp_score = None
        else:
            epp_score = epp_scoring_function(
                pts_source_normed, pts_target_normed, essen_mtx, self.norm_threshold, nfrm, to_compute_pair, self.epp_score
            )

        pts_prj_source, pts_prj_target, depth_prj, depth_prj_viewj = torch.split(prj_corres_depth, [2, 2, 1, 1], dim=2)
        pts_prj_source, pts_prj_target = kornia.geometry.conversions.convert_points_to_homogeneous(pts_prj_source), kornia.geometry.conversions.convert_points_to_homogeneous(pts_prj_target)
        intr_ = intr.view([1, 1, 1, 3, 3])
        fundm_mtx = intr_.inverse().transpose(-1, -2) @ essen_mtx @ intr_.inverse()
        pure_rotation = intr_ @ Rc @ intr_.inverse()

        intrinsic_t = intr_ @ ntc
        M = intr_ @ Rc @ intr_.inverse()

        if self.skip_prj_votes:
            prj_score =  None
        else:
            prj_score = prj_scoring(
                pts_prj_source, pts_prj_target, depth_prj, fundm_mtx, Rc, ntc, pure_rotation, intrinsic_t, M, intr_,
                self.nablate_prj, self.max_res_scale, self.prj_th, nfrm, to_compute_pair, self.prj_score
            )

        # Compute 3D consistence
        depth_embedding = torch.cat([self.depth_embedding_root, self.depth_embedding_rest])
        depth_prj_viewi_opted = depth_prj * depth_embedding[index_frmi].view([npair, 1, 1])
        depth_prj_viewj_opted = depth_prj_viewj * depth_embedding[index_frmj].view([npair, 1, 1])
        if self.skip_c3D_votes:
            c3D_score = None
        else:
            c3D_score = c3D_scoring(
                pts_prj_source, pts_prj_target, depth_prj_viewi_opted, depth_prj_viewj_opted, Rc, ntc, intr_,
                self.nablate_prj, self.max_res_scale, self.c3D_th
            )

        # dr = time.time() - st
        # print("Computing Epipolar and Projection Score Vote takes %.3f seconds" % dr)

        depth_ba = torch.cat([self.depth_embedding_root.data, self.depth_embedding_rest.data]).view([nfrm, 1]).repeat([1, nenum])
        depth_ba = depth_ba.view([nfrm, nenum, 1, 1])
        R_ba, nt_ba, scale_ba = divide_into_R_nt_scale(pose_all_candidates, self.eps)

        # Bundle Adjustment without projection constraints
        poses_opt, depth_opt = self.ba_wo_triangulation(
            R_ba, nt_ba, scale_ba, depth_ba, epp_score, prj_score, c3D_score, theta, index_frmi, index_frmj, intr.view([3, 3]),
            pose_all_candidates, pts_source_normed, pts_target_normed, pts_prj_source, pts_prj_target, depth_prj, depth_prj_viewj, optroot
        )

        # def scatter_plot(st_pos, ed_pos, rgb):
        #     import matplotlib.pyplot as plt
        #     import PIL.Image as Image
        #     w, h = rgb.size
        #
        #     stx, sty = st_pos
        #     stx, sty = (stx + 1) / 2 * w, (sty + 1) / 2 * h
        #     edx, edy = ed_pos
        #     edx, edy = (edx + 1) / 2 * w, (edy + 1) / 2 * h
        #
        #     # fig, ax = plt.subplots()
        #     plt.figure(figsize=(8, 16))
        #     plt.imshow(rgb)
        #     # if corres_gt is not None:
        #     #     plt.scatter(corres_gt[:, 0], corres_gt[:, 1], s=5.0, c=color)
        #
        #     plt.scatter(stx, sty, s=10.0, c='g')
        #     plt.scatter(edx, edy, s=10.0, c='c')
        #     plt.xlim([0, w])
        #     plt.ylim([h, 0])
        #     plt.axis("off")
        #     plt.savefig("tmp.png", format='png', bbox_inches='tight', pad_inches=0, dpi=800)
        #     rgb = Image.open("tmp.png").resize((int(2*w), int(2*h)))
        #     os.remove("tmp.png")
        #     plt.close()
        #     return Image.fromarray(np.array(rgb)[:, :, 0:3])
        #
        # def padding_border(fig):
        #     import PIL.Image as Image
        #     padding = 1
        #     fig = np.array(fig)
        #     h, w, _ = fig.shape
        #     fig[0:padding, :, :] = 255
        #     fig[h - padding:h, :, :] = 255
        #     fig[:, 0:padding, :] = 255
        #     fig[:, w - padding:w, :] = 255
        #     return Image.fromarray(fig)
        #
        # vls_idx = 0
        # for i in range(20):
        #     fig = tensor2disp(prj_score[i, vls_idx].view([1, 1, 100, 200]), vmax=1500)
        #     fig = scatter_plot(rec_pos[0][i], rec_pos[1999][i], fig)
        #     a = 1
        #     padding_border(fig).save(
        #         os.path.join('/home/shengjie/Documents/two2multiview/sparf/docs/ba_vls/{}.jpg'.format(str(i)))
        #     )

        if noreroot:
            rescale = depth_opt[0].squeeze().item()
            depth_opt = depth_opt / rescale
            poses_opt[:, :, 3:4] = poses_opt[:, :, 3:4] / rescale
            poses_opt = padding_pose(poses_opt)
            poses_opt = poses_opt @ poses_opt[0:1].inverse()
            poses_opt = poses_opt[:, 0:3, :]

        depth_gt = data_dict['depth_gt']
        bf_eppinlier, bf_prjinlier, bf_3DCinlier = \
            self.global_eppinlier(), self.global_prjinlier(), self.global_3DCinlier()
        af_eppinlier, af_prjinlier, af_3DCinlier = \
            self.global_eppinlier(poses_opt), self.global_prjinlier(poses_opt, depth_opt), self.global_3DCinlier(poses_opt, depth_opt)

        bf_score, af_score = 0, 0
        if not self.skip_epp_votes:
            bf_score, af_score = bf_score + bf_eppinlier, af_score + af_eppinlier
        if not self.skip_prj_votes:
            bf_score, af_score = bf_score + bf_prjinlier, af_score + af_prjinlier
        if not self.skip_c3D_votes:
            bf_score, af_score = bf_score + bf_3DCinlier * self.c3D_w, af_score + af_3DCinlier * self.c3D_w

        bf_posinlier, corres_bs = self.global_poseprj_inlier(w2c_poses_gt, depth_gt)
        af_posinlier, _ = self.global_poseprj_inlier(w2c_poses_gt, depth_gt, poses_opt)
        depth_adj_gt = list()
        for i in range(nfrm):
            selector = data_dict['depth_gt'][i] > 0
            scalar = torch.median(data_dict['depth_gt'][i][selector]) / torch.median(data_dict['depth_est'][i][selector])
            depth_adj_gt.append(scalar)
        depth_adj_gt = torch.stack(depth_adj_gt).view([nfrm, 1, 1])
        gt_eppinlier, gt_prjinlier, gt_3DCinlier = \
            self.global_eppinlier(w2c_poses_gt), self.global_prjinlier(w2c_poses_gt, depth_adj_gt), self.global_3DCinlier(w2c_poses_gt, depth_adj_gt)
        gt_posinlier, _ = self.global_poseprj_inlier(w2c_poses_gt, depth_gt, w2c_poses_gt)
        print("Epp Inliers, Before / After / Gt Inlier: %f / %f / %f" % (bf_eppinlier, af_eppinlier, gt_eppinlier))
        print("Prj Inliers, Before / After / Gt Inlier: %d / %d / %d" % (bf_prjinlier, af_prjinlier, gt_prjinlier))
        print("3DC Inliers, Before / After / Gt Inlier: %d / %d / %d" % (bf_3DCinlier, af_3DCinlier, gt_3DCinlier))
        print("Pos Inliers, Before / After / Gt Inlier / Corres Bs: %d / %d / %d / %d" % (bf_posinlier, af_posinlier, gt_posinlier, corres_bs))
        print("Before / After Scores: %.1f / %.1f" % (bf_score, af_score))
        statistics = {
            "epp_bf": bf_eppinlier,
            "epp_af": af_eppinlier,
            "epp_gt": gt_eppinlier,
            "prj_bf": bf_prjinlier,
            "prj_af": af_prjinlier,
            "prj_gt": gt_prjinlier,
            "3DC_bf": bf_3DCinlier,
            "3DC_af": af_3DCinlier,
            "3DC_gt": gt_3DCinlier,
            "pos_bf": bf_posinlier,
            "pos_af": af_posinlier,
            "pos_gt": gt_posinlier,
            "pos_bs": corres_bs
        }
        update = af_score > self.max_count + 5 # add small padding to prevent numerical noise

        self.document_compare_embedding(document=True, epp_score=epp_score, prj_score=prj_score, root=current_root)

        if update:
            self.max_count = af_score
            self.pose_embedding.data = pose_to_d10(poses_opt)
            self.depth_embedding_root.data = depth_opt[0:1, 0, 0]
            self.depth_embedding_rest.data = depth_opt[1::, 0, 0]

        return update, statistics

    def fill_wt_gt_poses(self, data_dict):
        w2c_poses_gt = padding_pose(data_dict['pose'])
        w2c_poses_gt = w2c_poses_gt @ w2c_poses_gt[0:1].inverse()
        updated_viewi_embedd = pose_to_d10(w2c_poses_gt)
        self.pose_embedding.data = updated_viewi_embedd

    def global_eppinlier(self, w2c_poses=None):

        if w2c_poses is None:
            w2c_poses = padding_pose(self.get_w2c_poses())
        else:
            w2c_poses = padding_pose(w2c_poses)

        intr = self.intr.view([1, 3, 3])
        inliers_sum, tot_sum = 0, 0
        for idx, (i, j) in enumerate(self.flow_pairs):
            corres_itoj = self.flatten_epp_evaluater[str([i, j])]
            pose_itoj = w2c_poses[j] @ w2c_poses[i].inverse()
            npose_itoj = torch.clone(pose_itoj)
            npose_itoj[0:3, 3] = npose_itoj[0:3, 3] / (npose_itoj[0:3, 3] ** 2).sum().sqrt()
            essential_itoj = cross_product_matrix(npose_itoj[0:3, 3]) @ npose_itoj[0:3, 0:3]

            corres_itoj_source, corres_itoj_target = torch.split(corres_itoj, 2, dim=1)
            corres_itoj_source = kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_source)
            corres_itoj_target = kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_target)

            npts = corres_itoj_source.shape[0]
            corres_itoj_source, corres_itoj_target = corres_itoj_source.view([npts, 3, 1]), corres_itoj_target.view([npts, 3, 1])
            corres_itoj_source, corres_itoj_target = intr.inverse() @ corres_itoj_source, intr.inverse() @ corres_itoj_target
            corres_itoj_source, corres_itoj_target = corres_itoj_source.view([npts, 3]), corres_itoj_target.view([npts, 3])

            npts = corres_itoj_source.shape[0]
            ex = essential_itoj.view([1, 3, 3]) @ corres_itoj_source.view([npts, 3, 1])
            xe = corres_itoj_target.view([npts, 1, 3]) @ essential_itoj.view([1, 3, 3])
            ex1, ex2, _ = torch.split(ex, 1, dim=1)
            xe1, xe2, _ = torch.split(xe, 1, dim=2)
            d = torch.sqrt(ex1 ** 2 + ex2 ** 2 + xe1 ** 2 + xe2 ** 2 + 1e-10)
            xex = corres_itoj_target.view([npts, 1, 3]) @ essential_itoj.view([1, 3, 3]) @ corres_itoj_source.view([npts, 3, 1])
            error = (xex / d).abs()
            inliers = error < self.norm_threshold
            inliers_sum += torch.sum(inliers)
            tot_sum += npts
        return inliers_sum.item() / tot_sum

    def global_prjinlier(self, w2c_poses=None, depth_opt=None):
        if w2c_poses is None:
            w2c_poses = padding_pose(self.get_w2c_poses())
        else:
            w2c_poses = padding_pose(w2c_poses)

        if depth_opt is None:
            depth_embedding = torch.clone(torch.cat([self.depth_embedding_root, self.depth_embedding_rest]))
        else:
            depth_embedding = depth_opt

        # depth_embedding = torch.cat([self.depth_embedding_root, self.depth_embedding_rest])
        intr = padding_pose(self.intr.view([1, 3, 3]))
        inliers_sum, tot_sum = 0, 0
        for idx, (i, j) in enumerate(self.flow_pairs):
            corres_itoj, depth_itoj, _ = self.flatten_prj_evaluator[str([i, j])]
            prj_itoj = intr @ w2c_poses[j] @ w2c_poses[i].inverse() @ intr.inverse()

            corres_itoj_source, corres_itoj_target = torch.split(corres_itoj, 2, dim=1)
            pts3d_itoj_source = kornia.geometry.conversions.convert_points_to_homogeneous(kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_source) * depth_itoj.unsqueeze(1) * depth_embedding[i].item())
            pts3d_itoj_source_prj = prj_itoj @ pts3d_itoj_source.unsqueeze(-1)
            pts2d_itoj_source_prj = kornia.geometry.conversions.convert_points_from_homogeneous(kornia.geometry.conversions.convert_points_from_homogeneous(pts3d_itoj_source_prj.squeeze(-1)))

            inliers_sum += torch.sum(torch.sqrt(torch.sum((pts2d_itoj_source_prj - corres_itoj_target) ** 2, dim=-1) + self.eps) < self.prj_th) * self.visibility_w[idx].item()
        return inliers_sum.item()

    def optimize_depth(self, depth_est):
        depth_embedding = torch.cat([self.depth_embedding_root, self.depth_embedding_rest])
        depth_embedding = depth_embedding.view([self.opt.train_sub, 1, 1])
        return depth_est * depth_embedding

    def optimize_depth_viewi(self, depth_viewi):
        depth_embedding = torch.cat([self.depth_embedding_root, self.depth_embedding_rest])
        depth_embedding = depth_embedding.view([self.opt.train_sub, 1])
        return depth_viewi * depth_embedding

    def optimize_depth_viewi_viewj(self, depth_viewi_viewj, pose_idx_viewj):
        depth_embedding = torch.cat([self.depth_embedding_root, self.depth_embedding_rest])
        depth_embedding = depth_embedding[pose_idx_viewj]
        nviewi, nviewj, _, _ = depth_viewi_viewj.shape
        depth_embedding = depth_embedding.view([nviewi, nviewj, 1, 1])
        return depth_viewi_viewj * depth_embedding

    @torch.enable_grad()
    def ba_wo_triangulation(
            self, R_ba, nt_ba, scale_ba, depth_ba, eppscore_ba, prjscore_ba, c3Dscore_ba, theta_ba, index_frmi, index_frmj, intr,
            posecandidates, pts_source_normed, pts_target_normed, pts_prj_source, pts_prj_target, depth_prj, depth_prj_viewj, optroot
    ):

        nfrm, nenum = R_ba.shape[0], R_ba.shape[1]

        R_ba_i, nt_ba_i = R_ba[index_frmi], nt_ba[index_frmi]
        R_ba_j, nt_ba_j = R_ba[index_frmj], nt_ba[index_frmj]
        R_ba_itoj = R_ba_j @ R_ba_i.transpose(-1, -2)
        if self.skip_epp_votes:
            eppscore_ba_f = None
        else:
            eppscore_ba_f = einops.rearrange(eppscore_ba, 'n k nba -> (n k) nba')
            eppscore_ba_f = eppscore_ba_f.unsqueeze(1).unsqueeze(1)
        if self.skip_prj_votes:
            prjscore_ba_f = None
        else:
            prjscore_ba_f = einops.rearrange(prjscore_ba, 'n k nba1 nba2 -> (n k) nba1 nba2')
            prjscore_ba_f = prjscore_ba_f.unsqueeze(1)

        if self.skip_c3D_votes:
            c3Dscore_ba_f = None
        else:
            c3Dscore_ba_f = einops.rearrange(c3Dscore_ba, 'n k nba1 nba2 -> (n k) nba1 nba2')
            c3Dscore_ba_f = c3Dscore_ba_f.unsqueeze(1)

        scale_ba, depth_ba = scale_ba.detach(), depth_ba.detach()
        scale_bas, depth_bas = torch.split(scale_ba, 1, dim=0), torch.split(depth_ba, 1, dim=0)
        scale_learn, depth_learn = list(), list()
        for i in range(len(scale_bas)):
            if i != optroot:
                scale_bas[i].requires_grad = True
                depth_bas[i].requires_grad = True
                scale_learn.append(scale_bas[i])
                depth_learn.append(depth_bas[i])
            else:
                scale_bas[i].requires_grad = False
                depth_bas[i].requires_grad = False

        st_lr, ed_lr = 1e-3, 1e-4
        if self.opt.mondepth_init == 'gtdepth':
            iterations, optimizer = 2000, torch.optim.Adam(scale_learn, lr=st_lr)
        else:
            iterations, optimizer = 2000, torch.optim.Adam(scale_learn + depth_learn, lr=st_lr)

        joint_sum_optimal, scale_optimal, depth_optimal = None, None, None

        # Iteration Part
        for iteration in range(iterations):
            scale_ba = torch.cat(scale_bas, dim=0)
            depth_ba = torch.cat(depth_bas, dim=0)
            scale_ba_i, scale_ba_j, depth_ba_i, depth_ba_j = scale_ba[index_frmi], scale_ba[index_frmj], depth_ba[index_frmi], depth_ba[index_frmj]
            t_ba_i, t_ba_j = nt_ba_i * scale_ba_i, nt_ba_j * scale_ba_j
            t_ba_itoj = t_ba_j - R_ba_itoj @ t_ba_i

            scale_ba_itoj = torch.sqrt(torch.sum(t_ba_itoj ** 2, dim=-2, keepdim=True) + self.eps)
            nt_ba_itoj = t_ba_itoj / scale_ba_itoj
            theta_ba_itoj = torch.clamp(torch.sum(nt_ba_itoj * nt_ba_j, dim=-2), min=-1+1e-3, max=1-1e-3)
            theta_ba_itoj = torch.arccos(theta_ba_itoj)

            joint_sum = 0

            if not self.skip_epp_votes:
                # Bilinear Interpolation on epp score
                theta_ba_itoj_sample = (theta_ba_itoj / theta_ba.view([int(nfrm*(nfrm-1)), nenum, 1]) - 0.5) * 2
                theta_ba_itoj_sample = torch.cat([theta_ba_itoj_sample, torch.zeros_like(theta_ba_itoj_sample)], dim=-1)
                theta_ba_itoj_samplef = einops.rearrange(theta_ba_itoj_sample, 'n k nba -> (n k) nba').unsqueeze(1).unsqueeze(1)
                theta_ba_itoj_samplef = torch.clamp(theta_ba_itoj_samplef, min=-1+1e-3, max=1-1e-3)
                eppscore_ba_f_s = torch.nn.functional.grid_sample(eppscore_ba_f, theta_ba_itoj_samplef, mode='bilinear', align_corners=True, padding_mode="zeros")
                eppscore_ba_f_s = eppscore_ba_f_s.view([int(nfrm*(nfrm-1)), nenum])
                eppscore_ba_f_sum = torch.sum(eppscore_ba_f_s, dim=0)
                joint_sum += eppscore_ba_f_sum / 5

            if not self.skip_prj_votes:
                # Bilinear Interpolation on prj score
                prj_ba_itoj_sample_y = (theta_ba_itoj / theta_ba.view([int(nfrm*(nfrm-1)), nenum, 1]) - 0.5) * 2
                prj_ba_itoj_sample_x_index = (scale_ba_itoj / depth_ba_i) / self.max_res_scale * self.nablate_prj - 0.5
                prj_ba_itoj_sample_x = (prj_ba_itoj_sample_x_index / (self.nablate_prj - 1) - 0.5) * 2
                prj_ba_itoj_sample = torch.cat([prj_ba_itoj_sample_x.squeeze(-1), prj_ba_itoj_sample_y], dim=-1)
                prj_ba_itoj_samplef = einops.rearrange(prj_ba_itoj_sample, 'n k nba -> (n k) nba').unsqueeze(1).unsqueeze(1)
                prjscore_ba_f_s = torch.nn.functional.grid_sample(prjscore_ba_f, prj_ba_itoj_samplef, mode='bilinear', align_corners=True, padding_mode="zeros")
                prjscore_ba_f_s = prjscore_ba_f_s.view([int(nfrm*(nfrm-1)), nenum])
                prjscore_ba_f_sum = torch.sum(prjscore_ba_f_s * self.visibility_w, dim=0)
                joint_sum += prjscore_ba_f_sum

            if not self.skip_c3D_votes:
                c3D_ba_itoj_sample_y = (theta_ba_itoj / theta_ba.view([int(nfrm*(nfrm-1)), nenum, 1]) - 0.5) * 2
                c3D_ba_itoj_sample_x_index = scale_ba_itoj / self.max_res_scale * self.nablate_prj - 0.5
                c3D_ba_itoj_sample_x = (c3D_ba_itoj_sample_x_index / (self.nablate_prj - 1) - 0.5) * 2
                c3D_ba_itoj_sample = torch.cat([c3D_ba_itoj_sample_x.squeeze(-1), c3D_ba_itoj_sample_y], dim=-1)
                c3D_ba_itoj_samplef = einops.rearrange(c3D_ba_itoj_sample, 'n k nba -> (n k) nba').unsqueeze(1).unsqueeze(1)
                c3Dscore_ba_f_s = torch.nn.functional.grid_sample(c3Dscore_ba_f, c3D_ba_itoj_samplef, mode='bilinear', align_corners=True, padding_mode="zeros")
                c3Dscore_ba_f_s = c3Dscore_ba_f_s.view([int(nfrm*(nfrm-1)), nenum])
                c3Dscore_ba_f_sum = torch.sum(c3Dscore_ba_f_s * self.visibility_w, dim=0)
                joint_sum += c3Dscore_ba_f_sum * self.c3D_w

            if joint_sum_optimal is None:
                joint_sum_optimal, scale_optimal, depth_optimal = joint_sum, torch.clone(scale_ba.detach()), torch.clone(depth_ba.detach())
            else:
                selector = (joint_sum_optimal > joint_sum).float()
                joint_sum_optimal = joint_sum * (1 - selector) + joint_sum_optimal * selector
                selector = selector.view([1, nenum, 1, 1])
                scale_optimal = scale_ba.detach() * (1 - selector) + scale_optimal * selector
                depth_optimal = depth_ba.detach() * (1 - selector) + depth_optimal * selector

            loss = joint_sum.mean()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

            for g in optimizer.param_groups:
                ratio = (iterations - iteration) / iterations
                g['lr'] = st_lr * ratio + ed_lr * (1 - ratio)

            # if np.mod(iteration, 100) == 0:
            #     print("Iteration %d, Loss %f, Max Epp / Prj Inliers %d / %d" % (iteration, loss.item(), eppscore_ba_f_sum.flatten().max().item(), prjscore_ba_f_sum.flatten().max().item()))

            """
            # Apply Random Check on Epipolar Score and Projection Score
            scale_ba_i, scale_ba_j = scale_ba[index_frmi], scale_ba[index_frmj]
            rnd_frm = np.random.randint(len(index_frmi))
            rndidx = np.random.randint(nenum)
            Ri, Rj = R_ba_i[rnd_frm, rndidx], R_ba_j[rnd_frm, rndidx]
            nti, ntj = nt_ba_i[rnd_frm, rndidx], nt_ba_j[rnd_frm, rndidx]
            scalei, scalej = scale_ba_i[rnd_frm, rndidx], scale_ba_j[rnd_frm, rndidx]
            theta = theta_ba[rnd_frm, rndidx]

            posei, posej = padding_pose(torch.cat([Ri, nti * scalei], dim=1).view([1, 3, 4])),  padding_pose(torch.cat([Rj, ntj * scalej], dim=1).view([1, 3, 4]))
            pose_i2j = posej @ posei.inverse()
            t_i2j = pose_i2j[:, 0:3, 3:4].view([3, 1])
            nt_i2j = t_i2j / torch.sqrt(torch.sum(t_i2j ** 2))

            theta_index = torch.clamp(torch.sum(nt_i2j * ntj), min=-1+self.eps, max=+1-self.eps)
            theta_index = torch.arccos(theta_index) / theta * (self.nablate_epp - 1)

            pose_i2j[:, 0:3, 3:4] = pose_i2j[:, 0:3, 3:4]
            intr_ = padding_pose(intr.view([1, 3, 3]))
            prj_index = torch.sqrt(torch.sum(t_i2j ** 2)).squeeze() / depth_ba_i[rnd_frm, rndidx] / self.max_res_scale * self.nablate_prj - 0.5

            pose_i2j_copy = torch.clone(pose_i2j)
            pose_i2j_copy[0, 0:3, 3:4] = pose_i2j_copy[0, 0:3, 3:4]
            prj_ck = intr_ @ pose_i2j_copy @ intr_.inverse()
            pts_prj_source_ck, pts_prj_target_ck, depth_prj_ck = pts_prj_source[rnd_frm], pts_prj_target[rnd_frm], depth_prj[rnd_frm]
            depth_prj_ck = depth_prj_ck * depth_ba_i[rnd_frm, rndidx]
            pts_prj_source_ck_prj = prj_ck @ kornia.geometry.convert_points_to_homogeneous(pts_prj_source_ck * depth_prj_ck).unsqueeze(-1)
            pts_prj_source_ck_prj = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts_prj_source_ck_prj.squeeze(-1)))
            prj_inlier = torch.sqrt(torch.sum((pts_prj_source_ck_prj - pts_prj_target_ck[:, 0:2]) ** 2, dim=-1))
            prj_inlier = torch.sum(prj_inlier < self.prj_th)
            print(prjscore_ba_f_s[rnd_frm, rndidx].item(), prj_inlier.item())

            pts_src, pts_tar, depth_src, depth_tar = pts_prj_source[rnd_frm], pts_prj_target[rnd_frm], depth_prj[rnd_frm], depth_prj_viewj[rnd_frm]
            pts_src_3D = kornia.geometry.convert_points_to_homogeneous(pts_src * depth_src * depth_ba_i[rnd_frm, rndidx].squeeze())
            pts_dst_3D = kornia.geometry.convert_points_to_homogeneous(pts_tar * depth_tar * depth_ba_j[rnd_frm, rndidx].squeeze())
            prj1 = pose_i2j_copy @ intr_.inverse()
            prj2 = intr_.inverse()
            pts_src_3D = prj1 @ pts_src_3D.unsqueeze(-1)
            pts_dst_3D = prj2 @ pts_dst_3D.unsqueeze(-1)
            inlier_3D = (torch.sqrt(torch.sum((pts_src_3D.squeeze()[:, 0:3] - pts_dst_3D.squeeze()[:, 0:3]) ** 2, dim=-1)) < self.c3D_th).sum()
            inlier_3D_ref = c3Dscore_ba_f_s[rnd_frm, rndidx]
            print(inlier_3D.item(), inlier_3D_ref.item())
            """

        max_score = torch.max(joint_sum_optimal)
        max_idx = torch.where(joint_sum_optimal == max_score)
        max_idx = max_idx[0].squeeze().cpu().numpy()
        max_idx = max_idx.item(0)
        scale_opt, depth_opt = scale_optimal[:, max_idx], depth_optimal[:, max_idx]
        R_opt, nt_opt = R_ba[:, max_idx], nt_ba[:, max_idx]
        poses_opt = torch.cat([R_opt, nt_opt * scale_opt], dim=-1).detach()
        poses_opt, depth_opt = poses_opt[:, 0:3, :], depth_opt
        return poses_opt, depth_opt

    @torch.no_grad()
    def global_3DCinlier(
            self, w2c_poses=None, depth_opt=None
    ):
        if w2c_poses is None:
            w2c_poses = padding_pose(self.get_w2c_poses())
        else:
            w2c_poses = padding_pose(w2c_poses)

        if depth_opt is None:
            depth_embedding = torch.clone(torch.cat([self.depth_embedding_root, self.depth_embedding_rest]))
        else:
            depth_embedding = depth_opt

        intr = padding_pose(self.intr.view([1, 3, 3]))
        inliers_sum, tot_sum = 0, 0
        for idx, (i, j) in enumerate(self.flow_pairs):
            corres_itoj, depth_viewi, depth_viewj = self.flatten_prj_evaluator[str([i, j])]
            prj_viewi2j = w2c_poses[j] @ w2c_poses[i].inverse() @ intr.inverse()
            prj_viewfrj = intr.inverse()

            corres_itoj_source, corres_itoj_target = torch.split(corres_itoj, 2, dim=1)
            pts3d_viewi2j = kornia.geometry.conversions.convert_points_to_homogeneous(kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_source) * depth_viewi.unsqueeze(1) * depth_embedding[i].item())
            pts3d_viewi2j = prj_viewi2j @ pts3d_viewi2j.unsqueeze(-1)

            pts3d_viewfrj = kornia.geometry.conversions.convert_points_to_homogeneous(kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_target) * depth_viewj.unsqueeze(1) * depth_embedding[j].item())
            pts3d_viewfrj = prj_viewfrj @ pts3d_viewfrj.unsqueeze(-1)

            pts3d_viewi2j, pts3d_viewfrj = pts3d_viewi2j[:, 0:3, 0], pts3d_viewfrj[:, 0:3, 0]
            inliers_sum += torch.sum(torch.sqrt(torch.sum((pts3d_viewi2j - pts3d_viewfrj) ** 2, dim=-1) + self.eps) < self.c3D_th) * self.visibility_w[idx].item()

        return inliers_sum.item()

    @torch.no_grad()
    def global_poseprj_inlier(
            self, w2c_poses_gt, depth_gt, w2c_poses=None
    ):
        if w2c_poses is None:
            w2c_poses = padding_pose(self.get_w2c_poses())
        else:
            w2c_poses = padding_pose(w2c_poses)

        intr = padding_pose(self.intr.view([1, 3, 3]))
        inliers_prj_sum, inliers_corres_sum, tot_sum = 0, 0, 0
        for idx, (i, j) in enumerate(self.flow_pairs):
            corres_itoj, _, _ = self.flatten_prj_evaluator[str([i, j])]
            corres_itoj_source, corres_itoj_target = torch.split(corres_itoj, 2, dim=1)
            depth_viewi_gt = depth_gt[i][corres_itoj_source[:, 1].long(), corres_itoj_source[:, 0].long()]
            valid = (depth_viewi_gt != 0)
            prj_viewi2j = intr @ w2c_poses[j] @ w2c_poses[i].inverse() @ intr.inverse()
            prj_viewi2j_gt = intr @ w2c_poses_gt[j] @ w2c_poses_gt[i].inverse() @ intr.inverse()

            corres_pts3D = kornia.geometry.conversions.convert_points_to_homogeneous(kornia.geometry.conversions.convert_points_to_homogeneous(corres_itoj_source) * depth_viewi_gt.unsqueeze(1))
            corres_prj = prj_viewi2j @ corres_pts3D.unsqueeze(-1)
            corres_prj_gt = prj_viewi2j_gt @ corres_pts3D.unsqueeze(-1)
            corres_prj = kornia.geometry.conversions.convert_points_from_homogeneous(kornia.geometry.conversions.convert_points_from_homogeneous(corres_prj.squeeze()))
            corres_prj_gt = kornia.geometry.conversions.convert_points_from_homogeneous(kornia.geometry.conversions.convert_points_from_homogeneous(corres_prj_gt.squeeze()))

            inliers_prj_sum += torch.sum(torch.sqrt(torch.sum((corres_prj - corres_prj_gt) ** 2, dim=-1) + self.eps)[valid] < self.prj_th) * self.visibility_w[idx].item()
            inliers_corres_sum += torch.sum(torch.sqrt(torch.sum((corres_itoj_target - corres_prj_gt) ** 2, dim=-1) + self.eps)[valid] < self.prj_th) * self.visibility_w[idx].item()

        return inliers_prj_sum.item(), inliers_corres_sum.item()

    @torch.no_grad()
    def document_compare_embedding(self, initialization=False, compare=False, document=False, epp_score=None, prj_score=None, root=None):
        if initialization:
            # Initialization or Change of Root
            self.pose_embedding_rec, self.depth_embedding_rec = None, None
            self.epp_score, self.prj_score = None, None
            self.last_root = -1
        elif compare:
            nfrm = len(self.pose_embedding)
            if self.pose_embedding_rec is None:
                assert self.pose_embedding_rec is None and self.depth_embedding_rec is None
                return torch.ones(int(nfrm * (nfrm - 1))) == 1
            elif root != self.last_root:
                assert root is not None and self.last_root is not None
                print("Root Switched, Recompute Everything....")
                self.epp_score, self.prj_score = None, None
                return torch.ones(int(nfrm * (nfrm - 1))) == 1
            else:
                change_pose = (self.pose_embedding_rec - self.pose_embedding).abs()
                change_pose = change_pose[:, 0:9] # Exclude Scale Change
                change_pose, _ = torch.max(change_pose, dim=-1)
                change_index_indicator = change_pose > 1e-6

                # Only Recompute Nodes related to changed index
                nfrm = self.pose_embedding.shape[0]
                to_compute_pair = torch.zeros(nfrm * (nfrm - 1)).cuda()
                cnt = 0
                for frmi in range(nfrm):
                    for frmj in range(nfrm):
                        if frmi == frmj:
                            continue
                        if change_index_indicator[frmi] or change_index_indicator[frmj]:
                            to_compute_pair[cnt] = 1
                        cnt += 1

                if nfrm * (nfrm - 1) < 10:
                    # Just compute everything given the small number of frames
                    to_compute_pair = torch.ones_like(to_compute_pair) == 1
                    self.prj_score = None

                print("Root Unchanged, Recompute %d / %d Pairs." % (torch.sum(to_compute_pair).item(), int(nfrm * (nfrm-1))))
                return to_compute_pair

        elif document:
            pose_embedding = self.pose_embedding.data
            depth_embedding = torch.cat([self.depth_embedding_root.data, self.depth_embedding_rest.data]).data
            self.pose_embedding_rec, self.depth_embedding_rec = torch.clone(pose_embedding.detach()), torch.clone(depth_embedding.detach())
            self.epp_score, self.prj_score = epp_score, prj_score
            self.last_root = root

    def reinitialization(self):
        self.max_count = 0
        self.document_compare_embedding(initialization=True)