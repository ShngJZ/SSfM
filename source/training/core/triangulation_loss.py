import time, secrets, os, copy
import einops
import imageio.plugins.freeimage
import matplotlib.pyplot as plt
import numpy as np
import torch
import kornia
import PIL.Image as Image
import tqdm

from kornia.geometry.epipolar.essential import cross_product_matrix
from easydict import EasyDict as edict
from typing import Any, Dict, Tuple
from source.training.core.base_losses import BaseLoss
from source.utils.camera import pose_inverse_4x4
from source.utils.helper import acquire_depth_range
from source.training.core.sampling_strategies import sample_rays
from source.utils.geometry.batched_geometry_utils import to_homogeneous, from_homogeneous, batch_backproject_to_3d, batch_project
from analysis.utils_evaldepth import compute_depth_errors
from analysis.utils_vls import tensor2rgb, tensor2disp

def extract_visibility(visibility_map, pixels_root, nview):
    visibility = list()
    for j in range(1, nview):
        visibility.append(visibility_map[str([0, j])][:, pixels_root[:, 1].long(), pixels_root[:, 0].long()])
    visibility = [torch.ones_like(visibility[-1])] + visibility
    visibility = torch.cat(visibility)
    return visibility

def padding_pose(poses_w2c):
    mh, mw = poses_w2c.shape[-2:]
    size = list(poses_w2c.shape[0:-2]) + [4, 4]
    poses_w2c_ = torch.zeros(size, device=poses_w2c.device)
    poses_w2c_[..., 3, 3] = 1.0
    poses_w2c_[..., 0:mh, 0:mw] = poses_w2c
    return poses_w2c_.contiguous()

def evaluate_depthmap_scale_improvement(renderer, data_dict):
    depth_est = data_dict['depth_est']
    depth_optimized = renderer.net.pose_net.optimize_depth(depth_est)
    depth_gt = data_dict['depth_gt']
    depth_metrics_est = compute_depth_errors(depth_gt[1::].flatten(), depth_est[1::].flatten())
    depth_metrics_opt = compute_depth_errors(depth_gt[1::].flatten(), depth_optimized[1::].flatten())
    renderer.logger.info(
        "Est / Opted, silog %.3f / %.3f, a05 %.3f / %.3f" %
        (depth_metrics_est[0], depth_metrics_opt[0], depth_metrics_est[-4], depth_metrics_opt[-4])
    )

class TriangulationLoss(BaseLoss):
    def __init__(self,
                 opt: Dict[str, Any],
                 nerf_net: torch.nn.Module,
                 corres_estimate_bundle: list[Any],
                 train_data: Dict[str, Any],
                 device: torch.device):
        super().__init__(device)
        self.opt = opt
        self.device = device
        self.net = nerf_net
        self.train_data = train_data

        self.corres_maps, self.flow_pairs, self.conf_maps, self.mask_valid_corr = corres_estimate_bundle
        self.flow_pairs_to_idx = dict()
        for idx, pair in enumerate(self.flow_pairs):
            self.flow_pairs_to_idx[str(pair)] = idx

        assert len(self.flow_pairs) == len(self.corres_maps)

        # Compute Visibility Map
        _, H, W = self.conf_maps[0].shape
        self.visibility_threshold = opt.loss_triangulation.visibility_threshold
        self.visibility_map, self.visibility_map_cat = self.compute_visibility_map(H, W)

        self.loss_tag = 'triangulation'
        self.dept_top_percent = opt.dept_top_percent

        self.w_corres = opt.loss_triangulation.w_corres
        self.w_corres_scale = opt.loss_triangulation.w_corres_scale

        if "reweight" in opt.loss_triangulation:
            self.reweight = opt.loss_triangulation.reweight
            self.reweight_w = opt.loss_triangulation.reweight_w
        else:
            self.reweight = False
            self.reweight_w = 1.0

        if "endupdate" in opt.loss_triangulation:
            # Only Select Pose Params after Scale Optimization
            self.endupdate = opt.loss_triangulation.endupdate
        else:
            self.endupdate = False

        if "focus" in opt.loss_triangulation:
            self.focus = opt.loss_triangulation.focus
        else:
            self.focus = False

        self.robust_counter = 2

    def ba_wo_triang_prjc(self, renderer, data_dict, skip_prj_votes, skip_c3D_votes, statistics_export=None):
        if skip_c3D_votes and not skip_prj_votes:
            renderer.logger.info("======== BA wo Triang on Prj Constraint ========")
        elif not skip_prj_votes and not skip_c3D_votes:
            renderer.logger.info("======== BA wo Triang on Prj and C3D Constraint ========")

        # Important to set to zero
        renderer.pose_net.skip_prj_votes, renderer.pose_net.skip_c3D_votes = skip_prj_votes, skip_c3D_votes

        # First Optimize camera scale and depth adjustment
        st = time.time()
        epoch, optimal_rec, statistics_all_epochs = 0, np.zeros(self.opt.train_sub), list()
        while True:
            if np.sum(optimal_rec) == self.opt.train_sub:
                break
            for frmi in range(self.opt.train_sub):
                while True:
                    update, statistics = renderer.pose_net.ransac_update_on_frame_i(
                        renderer=renderer, frmi=frmi, nfrm=renderer.settings.train_sub,
                        data_dict=data_dict, seed=0, foreval=False, noreroot=renderer.settings.noreroot
                    )
                    statistics_all_epochs.append(statistics)
                    if update:
                        statistics_export = statistics
                        epoch = epoch + 1
                        renderer.logger.info("Root Frm %d, Epoch %d Finished, Updated" % (frmi, epoch))
                        # Pose updated, Optimal Condition changed
                        for ii in range(self.opt.train_sub):
                            optimal_rec[ii] = 0
                    else:
                        renderer.logger.info("Root Frm %d, No Update" % frmi)
                        optimal_rec[frmi] = 1
                        break
                if np.sum(optimal_rec) == self.opt.train_sub:
                    break
        dr = time.time() - st
        statistics_export['duration'] = dr
        return statistics_export, statistics_all_epochs

    @torch.no_grad()
    def multiview_bundle_adjustment_pose(self, renderer, data_dict):

        # renderer.pose_net.fill_wt_gt_poses(data_dict)
        pose_results = renderer.evaluate_poses(renderer.settings)
        error_deg_before = max(pose_results['error_R_rel_deg'], pose_results['error_t_rel_deg'])

        assert self.opt.stage == 1
        if renderer.pose_net.skip_prj_votes and (not renderer.pose_net.skip_c3D_votes):
            statistics_export = self.ba_wo_triang_prjc(renderer, data_dict, skip_prj_votes=True, skip_c3D_votes=False, statistics_export=None)
        elif (not renderer.pose_net.skip_prj_votes) and renderer.pose_net.skip_c3D_votes:
            statistics_export = self.ba_wo_triang_prjc(renderer, data_dict, skip_prj_votes=False, skip_c3D_votes=True, statistics_export=None)
        else:
            raise NotImplementedError()

        torch.cuda.synchronize()
        pose_results = renderer.evaluate_poses(renderer.settings)
        error_deg_after = max(pose_results['error_R_rel_deg'], pose_results['error_t_rel_deg'])
        renderer.logger.info("Optimization Finished: Before / After Optimization: %f / %f" % (error_deg_before, error_deg_after))

        # Wrap Up
        evaluate_depthmap_scale_improvement(renderer, data_dict)
        renderer.export_pose_params_for_evaluation("pose_optimized")
        renderer.export_corres_from_pose_for_evaluation()

        import pickle
        statistics_path = os.path.join(renderer.writer.log_dir, 'statistics.pkl')
        with open(statistics_path, 'wb') as handle:
            pickle.dump(statistics_export[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
        statistics_path = os.path.join(renderer.writer.log_dir, 'statistics_all_epochs.pkl')
        with open(statistics_path, 'wb') as handle:
            pickle.dump(statistics_export[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def compute_visibility_map(self, H, W):
        visibility_map, visibility_map_cat = dict(), list()
        for idx, (i, j) in enumerate(self.flow_pairs):
            corresx, corresy = torch.split(self.corres_maps[idx], 1, dim=0)
            corresx, corresy = corresx.squeeze(), corresy.squeeze()
            visibility_map[str([i, j])] = (self.conf_maps[idx] > self.visibility_threshold) * \
                                     (corresx > 0) * (corresx < W - 1) * (corresy > 0) * (corresy < H - 1)
            visibility_map[str([i, j])] = visibility_map[str([i, j])].float()
            visibility_map_cat.append(visibility_map[str([i, j])].float())
        visibility_map_cat = torch.cat(visibility_map_cat, dim=0)
        return visibility_map, visibility_map_cat

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any],
                     iteration: int, mode: str=None, plot: bool=False, renderer=Any, do_log=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # Output
        state_dict, loss = dict(), 0

        # ++++++++ Prepare Pose and Others ++++++++
        _, _, H, W = data_dict['image'].shape
        idx_root = 0

        # Render Pose
        poses_w2c = renderer.net.pose_net.get_w2c_poses(detach_npose=True, detach_scale=True).detach()
        data_dict.poses_w2c = poses_w2c
        poses_w2c = padding_pose(poses_w2c)
        intirnsic44 = padding_pose(data_dict.intr)

        assert (renderer.settings.sampled_fraction_in_center == 0.0) and ('depth_est' in data_dict.keys())

        # Sampling
        pixels_root, _ = sample_rays(
            H, W, nbr=renderer.settings.nerf.rand_rays,
            fraction_in_center=renderer.settings.sampled_fraction_in_center,
        )
        pixels_root, depth_range = pixels_root.cuda(), acquire_depth_range(renderer.settings, data_dict)

        # NeRF Render over mult-view
        depth_viewj, pts2d_in_view_j = self.render_from_multiview(
            data_dict, renderer, poses_w2c, intirnsic44, pixels_root, depth_range, idx_root, H, W
        )

        # Compute Depth Loss
        triang_depth_loss, _ = self.compute_depth_loss(
            renderer, depth_viewj, pts2d_in_view_j, pixels_root, data_dict, H, W, renderer.settings.train_sub
        )

        # Compute Correspondence Loss
        triang_corres_loss, _, _, _, _ = self.compute_pose_scale_loss(
            opt, data_dict, renderer, intirnsic44, poses_w2c, pts2d_in_view_j, depth_viewj, depth_range, H, W, renderer.settings.train_sub
        )
        loss = triang_depth_loss * (1 - self.w_corres) + triang_corres_loss * self.w_corres_scale * self.w_corres

        # Evaluaiton on Statistics
        if do_log:
            with torch.no_grad():
                triang_scale_loss, triangulation_selector, correspondence_valid_pair, correspondence_estimator_pair, pts2d_in_ref_pair = self.compute_pose_scale_loss(
                    opt, data_dict, renderer, intirnsic44, poses_w2c, pts2d_in_view_j.detach(), depth_viewj.detach(), depth_range, H, W, renderer.settings.train_sub
                )
                correspondence_valid_pair = correspondence_valid_pair > 0.5
                state_dict['triang_density'] = (torch.sum(correspondence_valid_pair, dim=[1, 2]) / len(triangulation_selector)).cpu().numpy().min()

                pixels_root_xx, pixels_root_yy = torch.split(pixels_root.long(), 1, dim=-1)
                pixels_root_xx, pixels_root_yy = pixels_root_xx.squeeze(-1), pixels_root_yy.squeeze(-1)
                gtdepth_sampled = data_dict['depth_gt'][0, pixels_root_yy, pixels_root_xx]
                monodepth_sampled = data_dict['depth_est'][0, pixels_root_yy, pixels_root_xx]
                monodepth_triang = depth_viewj[0]
                silog_a05_mono = compute_depth_errors(gtdepth_sampled[triangulation_selector], monodepth_sampled[triangulation_selector])
                silog_a05_tria = compute_depth_errors(gtdepth_sampled[triangulation_selector], monodepth_triang[triangulation_selector])

                depth_viewj_frommono, pts2d_in_view_j_fromono, _ = self.render_from_squeezeed_multiview(
                    opt, data_dict, output_dict, renderer, poses_w2c, intirnsic44, pixels_root, depth_range, idx_root, H, W, frommono=True
                )
                _, _, _, _, pts2d_in_ref_pair_frommono = self.compute_pose_scale_loss(
                    opt, data_dict, renderer, intirnsic44, poses_w2c, pts2d_in_view_j_fromono, depth_viewj_frommono, depth_range, H, W, renderer.settings.train_sub
                    )

                gt_crres, est_corres, triang_corres, mono_corres = list(), list(), list(), list()
                for viewj in range(1, renderer.settings.train_sub):
                    corres_sel = correspondence_valid_pair[self.flow_pairs_to_idx[str([0, viewj])]] == 1
                    corres_sel = corres_sel.squeeze(1)
                    c_gt_crres = data_dict['corres_gts_root2others'][int(viewj - 1)][:, pixels_root_yy, pixels_root_xx].T
                    c_gt_crres = c_gt_crres[corres_sel, :]

                    c_est_corres = correspondence_estimator_pair[self.flow_pairs_to_idx[str([0, viewj])]]
                    c_est_corres = c_est_corres[corres_sel, :]

                    c_triang_corres = pts2d_in_ref_pair[self.flow_pairs_to_idx[str([0, viewj])]]
                    c_triang_corres = c_triang_corres[corres_sel, :]

                    c_mono_corres = pts2d_in_ref_pair_frommono[self.flow_pairs_to_idx[str([0, viewj])]]
                    c_mono_corres = c_mono_corres[corres_sel, :]

                    gt_crres.append(c_gt_crres)
                    est_corres.append(c_est_corres)
                    triang_corres.append(c_triang_corres)
                    mono_corres.append(c_mono_corres)

            gt_crres, est_corres, triang_corres, mono_corres = torch.cat(gt_crres, dim=0), torch.cat(est_corres, dim=0), torch.cat(triang_corres, dim=0), torch.cat(mono_corres, dim=0)
            px1_est = (torch.sum((gt_crres - est_corres) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_nerf = (torch.sum((gt_crres - triang_corres) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_mono = (torch.sum((gt_crres - mono_corres) ** 2, dim=1).sqrt() < 1).float().mean().item()
            eval_dict = {
                'px1_est': px1_est,
                'px1_nerf': px1_nerf,
                'px1_mono': px1_mono,
                'a05_mono': silog_a05_mono[-4],
                'a05_tria': silog_a05_tria[-4],
                'silog_mono': silog_a05_mono[0],
                'silog_tria': silog_a05_tria[0],
            }
            state_dict.update(eval_dict)
        return {self.loss_tag: loss}, state_dict, {}

    def triangulation_check_depth(self, pts3d_in_w_from_ref, depth_viewj):
        pts3d_diff = pts3d_in_w_from_ref[1::] - pts3d_in_w_from_ref[0:1]
        pts3d_diff = torch.sqrt(torch.sum(pts3d_diff ** 2, dim=-1) + 1e-10) / depth_viewj[0:1]
        pts3d_diff_for_sort = torch.mean(pts3d_diff, dim=0)
        val, idx = torch.sort(pts3d_diff_for_sort)

        threshold = val[int(len(val) * self.dept_top_percent)]
        triangulation_selector = pts3d_diff_for_sort < threshold
        return triangulation_selector

    def compute_pose_scale_loss(
            self, opt, data_dict, renderer, intirnsic44, poses_w2c, pts2d_in_view_j, depth_viewj, depth_range, H, W, nview
    ):
        # Compute 3D point
        pts3d_in_w_from_ref = batch_backproject_to_3d(
            kpi=pts2d_in_view_j,
            di=depth_viewj,
            Ki=intirnsic44[:, :3, :3].contiguous(),
            T_itoj=poses_w2c.inverse()
        )

        triangulation_selector = self.triangulation_check_depth(pts3d_in_w_from_ref, depth_viewj)

        # Project to other view
        prj_matrix = (intirnsic44 @ poses_w2c).unsqueeze(0)
        pts2d_in_ref = to_homogeneous(pts3d_in_w_from_ref.unsqueeze(1))
        pts2d_in_ref = pts2d_in_ref @ prj_matrix.transpose(-1, -2)
        pts2d_in_ref = from_homogeneous(from_homogeneous(pts2d_in_ref))

        # Sample for Correspondence
        pts2d_sample_pair, correspondence_nerf_pair, correspondence_pair = list(), list(), list()
        valid_corres_pair = list()
        for i in range(nview):
            for j in range(nview):
                if i != j:
                    pts2d_sample_pair.append(pts2d_in_view_j[i])
                    correspondence_nerf_pair.append(pts2d_in_ref[i, j])
                    correspondence_pair.append(self.corres_maps[self.flow_pairs_to_idx[str([i, j])]])

                    if self.reweight:
                        conf_map, vis_map = self.conf_maps[self.flow_pairs_to_idx[str([i, j])]], self.visibility_map[str([i, j])]
                        conf_map = conf_map ** self.reweight_w
                        valid_corres_pair.append(conf_map * vis_map)
                    else:
                        valid_corres_pair.append(self.visibility_map[str([i, j])])

        pts2d_sample_pair, pts2d_in_ref_pair, correspondence_pair = torch.stack(pts2d_sample_pair, dim=0).contiguous(), torch.stack(correspondence_nerf_pair, dim=0).contiguous(), torch.stack(correspondence_pair, dim=0).contiguous()
        valid_corres_pair = torch.stack(valid_corres_pair, dim=0).float().contiguous()

        # Sample the correspondence for loss
        samplexx, sampleyy = torch.split(pts2d_sample_pair.unsqueeze(2), 1, dim=3)
        samplexx, sampleyy = (samplexx / (W - 1) - 0.5) * 2, (sampleyy / (H - 1) - 0.5) * 2
        sample_normed = torch.cat([samplexx, sampleyy], dim=-1)

        correspondence_estimator_pair = torch.nn.functional.grid_sample(
            correspondence_pair, sample_normed, mode='bilinear', align_corners=True)
        correspondence_estimator_pair = correspondence_estimator_pair.squeeze(-1).permute([0, 2, 1])

        correspondence_valid_pair = torch.nn.functional.grid_sample(
            valid_corres_pair, sample_normed, mode='nearest', align_corners=True)
        correspondence_valid_pair = correspondence_valid_pair.squeeze(-1).permute([0, 2, 1])
        correspondence_valid_pair = correspondence_valid_pair * triangulation_selector.unsqueeze(0).unsqueeze(-1)

        triang_pose_loss = ((correspondence_estimator_pair - pts2d_in_ref_pair).abs() * correspondence_valid_pair).sum() / (correspondence_valid_pair.sum() + 1e-10)
        triang_pose_loss = triang_pose_loss / torch.sqrt(intirnsic44[0, 0, 0] ** 2 + intirnsic44[0, 1, 1] ** 2).item()

        # rndb, rndc = np.random.randint(0, correspondence_valid_pair.shape[0]), np.random.randint(0, correspondence_valid_pair.shape[1])
        # while correspondence_valid_pair[rndb, rndc, 0].item() == 0:
        #     rndb, rndc = np.random.randint(0, correspondence_valid_pair.shape[0]), np.random.randint(0, correspondence_valid_pair.shape[1])
        # imgsrcidx, imgdstidx = self.flow_pairs[rndb]
        # imgsrc, imgdst = tensor2rgb(data_dict['image'], viewind=imgsrcidx), tensor2rgb(data_dict['image'], viewind=imgdstidx)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # fig, axs = plt.subplots(2)
        # axs[0].imshow(imgsrc), axs[1].imshow(imgdst)
        # axs[0].scatter(pts2d_sample_pair[rndb, rndc, 0].item(), pts2d_sample_pair[rndb, rndc, 1].item())
        # axs[1].scatter(pts2d_in_ref_pair[rndb, rndc, 0].item(), pts2d_in_ref_pair[rndb, rndc, 1].item())
        # axs[1].scatter(correspondence_estimator_pair[rndb, rndc, 0].item(), correspondence_estimator_pair[rndb, rndc, 1].item())
        # plt.show()

        return triang_pose_loss, triangulation_selector, correspondence_valid_pair, correspondence_estimator_pair, pts2d_in_ref_pair

    def compute_depth_loss(self, renderer, depth_viewj, pts2d_in_view_j, pixels_root, data_dict, H, W, nview, usegt=False):
        sample_xx, sample_yy = torch.split(pts2d_in_view_j, 1, dim=2)
        sample_xx, sample_yy = (sample_xx / (W - 1) - 0.5) * 2, (sample_yy / (H - 1) - 0.5) * 2
        sample_pxls_viewi_normed = torch.cat([sample_xx, sample_yy], dim=-1)

        if usegt:
            depth_est = data_dict['depth_gt']
        else:
            depth_est = renderer.net.pose_net.optimize_depth(data_dict['depth_est']).detach()

        depth_sparse = torch.nn.functional.grid_sample(
            depth_est.view([nview,  1, H, W]), sample_pxls_viewi_normed.unsqueeze(1), mode='bilinear', align_corners=True)
        depth_sparse = depth_sparse.squeeze()

        visibility = extract_visibility(self.visibility_map, pixels_root, nview)
        visibility = visibility * (sample_xx.squeeze(2) >= -1) * (sample_xx.squeeze(2) <= 1) * (sample_yy.squeeze(2) >= -1) * (sample_yy.squeeze(2) <=1) * (depth_sparse > 0)
        visibility = visibility.float()

        triang_depth_loss = (depth_sparse - depth_viewj).abs()
        triang_depth_loss = torch.sum(triang_depth_loss * visibility) / (visibility.sum() + 1e-5)
        return triang_depth_loss, depth_sparse

    def render_from_multiview(self, data_dict, renderer, poses_w2c, intirnsic44, pixels_root, depth_range, idx_root, H, W):
        with torch.no_grad():
            # Render from Root Frame
            ret_root = renderer.net.render_image_at_specific_pose_and_rays(
                renderer.settings, data_dict, depth_range, poses_w2c[idx_root, :3].detach(),
                intr=intirnsic44[idx_root, :3, :3], H=H, W=W, pixels=pixels_root, mode='train', iter=renderer.iteration
            )
            depth_root = ret_root.depth.squeeze(0).squeeze(-1)

            # Project to Support Frames
            pts3d_in_w_from_ref = batch_backproject_to_3d(
                kpi=pixels_root.unsqueeze(0),
                di=depth_root.unsqueeze(0),
                Ki=intirnsic44[idx_root, :3, :3].unsqueeze(0),
                T_itoj=poses_w2c[idx_root].unsqueeze(0)
            )
            pts2d_in_view_j, depth_in_view_j = batch_project(
                pts3d_in_w_from_ref.expand([len(poses_w2c), -1, -1]), T_itoj=poses_w2c, Kj=intirnsic44[:, 0:3, 0:3], return_depth=True
            )

        # Rendering, require grad
        ret_viewj = renderer.net.render_image_at_specific_pose_and_rays(
            renderer.settings, data_dict, depth_range, poses_w2c[:, :3].detach(),
            intr=intirnsic44[:, :3, :3], H=H, W=W, pixels=pts2d_in_view_j, mode='train', iter=renderer.iteration
        )
        depth_viewj = ret_viewj['depth'].squeeze(-1)

        return depth_viewj, pts2d_in_view_j

    def render_from_squeezeed_multiview(
            self, opt, data_dict, output_dict, renderer, poses_w2c, intirnsic44, pixels_root, depth_range, idx_root, H, W, frommono=False):
        if frommono:
            squeezed_nerf = data_dict['depth_est']
        else:
            squeezed_nerf = renderer.net.squeeze_nerf(data_dict)

        # Render from Root Frame
        depth_root = squeezed_nerf[idx_root, pixels_root[:, 1].long(), pixels_root[:, 0].long()]

        # Project to Support Frames
        pts3d_in_w_from_ref = batch_backproject_to_3d(
            kpi=pixels_root.unsqueeze(0),
            di=depth_root.unsqueeze(0),
            Ki=intirnsic44[idx_root, :3, :3].unsqueeze(0),
            T_itoj=poses_w2c[idx_root].unsqueeze(0)
        )
        pts2d_in_view_j, depth_in_view_j = batch_project(
            pts3d_in_w_from_ref.expand([len(poses_w2c), -1, -1]), T_itoj=poses_w2c, Kj=intirnsic44[:, 0:3, 0:3], return_depth=True
        )

        samplexx, sampleyy = torch.split(pts2d_in_view_j.unsqueeze(2), 1, dim=3)
        samplexx, sampleyy = (samplexx / (W - 1) - 0.5) * 2, (sampleyy / (H - 1) - 0.5) * 2
        sample_normed = torch.cat([samplexx, sampleyy], dim=-1)
        depth_viewj = torch.nn.functional.grid_sample(
            squeezed_nerf.unsqueeze(1), sample_normed, mode='bilinear', align_corners=True)
        depth_viewj = depth_viewj.squeeze()

        ret_viewj = dict()
        ret_viewj['pts2d_in_multiview'] = pts2d_in_view_j
        ret_viewj['depth'] = depth_viewj
        ret_viewj['visibility_per_sampled'] = torch.ones_like(depth_viewj) == 1
        return depth_viewj, pts2d_in_view_j, ret_viewj

    def acq_accum_corr_corect_loc(self, idx_img_rendered):
        accum_corr_corect_loc = 0
        for pair in self.flow_pairs:
            if pair[1] == idx_img_rendered:
                accum_corr_corect_loc += self.mask_valid_corr[pair[0]].float()
        return accum_corr_corect_loc

    @torch.no_grad()
    def inference_sparse_depth(self, renderer, data_dict):
        _, _, H, W = data_dict['image'].shape
        idx_root, opt, nview, dummy_iteration = 0, renderer.settings, renderer.settings.train_sub, 10000

        # Render Pose
        poses_w2c = renderer.net.get_w2c_pose(renderer.settings, data_dict, mode='eval')  # is it world to camera
        poses_w2c = padding_pose(poses_w2c)
        intirnsic44 = padding_pose(data_dict.intr)

        """
        # Visualize Difference
        fig_all = list()
        for idx_view in range(nview):
            rendered = renderer.net.render_image_at_specific_rays(opt, data_dict, img_idx=idx_view, iter=dummy_iteration, mode="eval")
            depth_nerf, depth_mono = rendered['depth'].view([1, 1, H, W]), data_dict['depth_est'][idx_view].view([1, 1, H, W])
            depth_diff = (depth_nerf - depth_mono) / depth_mono
            fig_depth_nerf, fig_depth_mono = tensor2disp(1/depth_nerf, vmax=1.0, viewind=0), tensor2disp(1/depth_mono, vmax=1.0, viewind=0)
            fig_depth_diff = tensor2grad(depth_diff, pos_bar=0.01, neg_bar=0.01, viewind=0)
            fig_rgb = tensor2rgb(data_dict['image'], viewind=idx_view)
            fig_all.append(np.concatenate(
                [np.array(fig_rgb), np.array(fig_depth_nerf), np.array(fig_depth_mono), np.array(fig_depth_diff)], axis=0
            ))
        fig_all = np.concatenate(fig_all, axis=1)
        Image.fromarray(fig_all).show()
        """

        # Sampling
        pixels_root, _ = sample_rays(
            H, W, nbr=renderer.settings.nerf.rand_rays,
            fraction_in_center=renderer.settings.sampled_fraction_in_center,
        )
        pixels_root, depth_range = pixels_root.cuda(), acquire_depth_range(renderer.settings, data_dict)
        depth_viewj_nerf, pts2d_in_view_j, _ = self.render_from_multiview(
            opt, data_dict, None, renderer, poses_w2c, intirnsic44, pixels_root, depth_range, idx_root, H, W
        )
        _, visibility, depth_viewj_mono = self.compute_depth_loss(
            renderer, depth_viewj_nerf, pts2d_in_view_j, pixels_root, data_dict, H, W, renderer.settings.train_sub
        )
        _, _, depth_viewj_gt = self.compute_depth_loss(
            renderer, depth_viewj_nerf, pts2d_in_view_j, pixels_root, data_dict, H, W, renderer.settings.train_sub, usegt=True
        )

        def backproject_to_2D(pts2d_in_view_j, depth_viewj, intrinsic, poses_w2c):
            intrinsic = intrinsic.view([1, 4, 4])
            prjM = intrinsic @ poses_w2c.inverse() @ intrinsic.inverse()
            pts3D_world = to_homogeneous(to_homogeneous(pts2d_in_view_j) * depth_viewj.unsqueeze(2)) @ prjM.transpose(-1, -2)
            pts3D_world = from_homogeneous(pts3D_world)
            pts2D_root = from_homogeneous(pts3D_world)
            return pts2D_root
        def compute_pts2D_error_matrix(pts3D_world, nview):
            error_matrix = ((pts3D_world.unsqueeze(0) - pts3D_world.unsqueeze(1)) ** 2 + 1e-10).sum(dim=-1, keepdim=True).sqrt()
            error_matrix = error_matrix
            error_matrix = error_matrix.sum(dim=[0, 1, 3]) / 2 / (nview - 1) / (nview - 1)
            return error_matrix

        pts3D_world_nerf = backproject_to_2D(pts2d_in_view_j, depth_viewj_nerf, intirnsic44[0], poses_w2c)
        pts3D_error_nerf = compute_pts2D_error_matrix(pts3D_world_nerf, nview)
        pts3D_world_mono = backproject_to_2D(pts2d_in_view_j, depth_viewj_mono, intirnsic44[0], poses_w2c)
        pts3D_error_mono = compute_pts2D_error_matrix(pts3D_world_mono, nview)
        pts3D_world_gt = backproject_to_2D(pts2d_in_view_j, depth_viewj_gt, intirnsic44[0], poses_w2c)
        pts3D_error_gt = compute_pts2D_error_matrix(pts3D_world_gt, nview)

        consist_improve = pts3D_error_nerf - pts3D_error_mono
        _, improve_idx = torch.sort(consist_improve)

        densities = np.linspace(0.02, 0.95, 200)
        silog_a05_monos, silog_a05_nerfs = list(), list()
        for density in densities:
            evaluate_idx = int(np.round(density * renderer.settings.nerf.rand_rays).item())

            evaluate_visibile = visibility[:, 0:evaluate_idx] == 1
            evaluate_gt = depth_viewj_gt[:, 0:evaluate_idx]
            evaluate_nerf = depth_viewj_nerf[:, 0:evaluate_idx]
            evaluate_mono = depth_viewj_mono[:, 0:evaluate_idx]

            silog_a05_mono = compute_depth_errors(evaluate_gt[evaluate_visibile], evaluate_mono[evaluate_visibile])
            silog_a05_nerf = compute_depth_errors(evaluate_gt[evaluate_visibile], evaluate_nerf[evaluate_visibile])

            silog_a05_monos.append(silog_a05_mono)
            silog_a05_nerfs.append(silog_a05_nerf)

        silog_a05_monos = np.stack(silog_a05_monos, axis=0)
        silog_a05_nerfs = np.stack(silog_a05_nerfs, axis=0)

        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(1, 2, figsize=(16, 16))
        axs[0].plot(densities, silog_a05_monos[:, 0])
        axs[0].plot(densities, silog_a05_nerfs[:, 0])
        axs[0].set_xlabel("Triang Th in Meter")
        axs[0].set_ylabel("Silog")
        axs[0].legend(['Mono', 'MultiView'])
        axs[0].set_title(opt.scene)

        axs[1].plot(densities, silog_a05_monos[:, -4])
        axs[1].plot(densities, silog_a05_nerfs[:, -4])
        axs[1].set_xlabel("Triang Th in Meter")
        axs[1].set_ylabel("A05")
        axs[1].legend(['Mono', 'MultiView'])
        axs[1].set_title(opt.scene)

        plt.show()

