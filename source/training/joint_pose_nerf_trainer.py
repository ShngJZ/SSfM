import copy
import numpy as np
import os, time, cv2
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from typing import Any, List, Tuple, Dict

import source.training.nerf_trainer as nerf
import source.utils.camera as camera
from source.models.poses_models.two_columns_scale_optdepth import FirstTwoColunmnsScalePoseOptDepthParameters
from source.training.engine.iter_based_trainer import get_log_string
from source.models.renderer import Graph
from source.utils.helper import InputPadder
from source.training.core.corres_loss import CorrespondencesPairRenderDepthAndGet3DPtsAndReproject
from source.training.core.triangulation_loss import TriangulationLoss

class CommonPoseEvaluation:

    def set_initial_poses(self, opt: Dict[str, Any]) -> Tuple[torch.Tensor, List[int], List[int]]:
        """ Define initial poses (to be optimized) """

        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(self.device)
        n_poses = len(self.train_data)
        valid_poses_idx = np.arange(start=0, stop=n_poses, step=1).tolist()
        index_images_excluded = []
        assert not opt.camera.optimize_relative_poses

        if 'sfm' in opt.camera.initial_pose:
            # initial poses obtained by COLMAP with different matchers
            # we all save it into a common directory, to make sure that if it was done before, it is not done again

            directory_save = os.path.join(self._base_save_dir, 'colmap_initial_results', self.settings.dataset)
            if self.settings.train_sub is not None and self.settings.train_sub != 0:
                directory_save = os.path.join(directory_save, 'subset_{}'.format(self.settings.train_sub))
            directory_save = os.path.join(directory_save, self.settings.scene)

            directory = directory_save
            os.makedirs(directory_save, exist_ok=True)

            if 'pdcnet' in opt.camera.initial_pose:
                st_colmap = time.time()
                initial_poses_w2c, valid_poses_idx, index_images_excluded = \
                    compute_sfm_pdcnet(opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
                dr = time.time() - st_colmap
                data_input = os.path.join(self.writer.log_dir, 'colmap_execute_time.pickle')
                file = open(data_input, 'wb')
                import pickle
                pickle.dump({'duration': dr}, file)
                file.close()
            else:
                raise ValueError

            initial_poses_w2c = initial_poses_w2c.to(self.device).float()
            initial_poses_w2c, ssim_est_gt_c2w = self.prealign_w2c_small_camera_systems\
                (opt, initial_poses_w2c[:, :3], pose_GT_w2c[:, :3])

        elif 'twoview' in opt.camera.initial_pose:
            def generate_keypair_list(n_views: int):
                """Generate list of possible exhaustive pairs, (Nx2). """
                pairs = []
                for i in range(1, n_views):
                    pairs.append([0, i])
                # pairs is N
                pairs = np.array(pairs)  # N x 2
                return torch.from_numpy(pairs.T)
            def twoview_pose_estimation(
                    correspondence,
                    confidence,
                    mask_valid_corr,
                    depthmap,
                    intrinsic,
                    gridpxls=None,
                    norm_threshold_adjuster=0.4,
                    minconf=0.8,
                    use_opencv=True,
                    use_gtscale=True,
                    gt_rel_pose=None,
                    rel_norm_threshold_adjuster=None
            ):
                """
                correspondence: 1 x 2 x H x W Correspondence,
                confidence: 1 x 1 x H x W Confidence in floating number, suggesting sampling probability,
                depthmap: 1 x 1 x H x W Depthmap,
                intrinsic: 3 x 3 intrinsic matrix,
                gridpxls: 1 x 2 x H x W griddle location, if provided, depending on different correspondence estimator convention,
                norm_threshold_adjuster: Threshold in inlier counting using Sampson distance, following Opencv convenction,
                minconf: minimal confidence threshold for correspondence estimator,
                use_opencv: use opencv two-view estimator,
                use_gtscale: use gt camera scale,
                gt_rel_pose: GT Relative Camera Pose to extract gt scale
                """
                import einops, kornia, cv2
                assert intrinsic.ndim == 2
                assert (intrinsic.shape[0] == 3) and (intrinsic.shape[1] == 3)
                assert correspondence.ndim == 4
                assert confidence.ndim == 4

                norm_threshold = norm_threshold_adjuster / intrinsic[:2, :2].abs().mean().item()

                if gridpxls is None:
                    # PDC Net Convention
                    _, _, H, W = correspondence.shape
                    gridxx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                    gridyy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                    gridxx = gridxx.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()
                    gridyy = gridyy.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()
                    gridpxls = torch.cat([gridxx, gridyy], dim=1)

                correspondencef = einops.rearrange(correspondence, 'b c h w -> (b h w) c')
                gridpxlsf = einops.rearrange(gridpxls, 'b c h w -> (b h w) c')
                confidencef = einops.rearrange(confidence.float(), 'b c h w -> (b h w) c').squeeze(-1)
                mask_valid_corrf = einops.rearrange(mask_valid_corr, 'b c h w -> (b h w) c').squeeze(-1)

                correspondencef, gridpxlsf, confidencef = correspondencef[mask_valid_corrf], gridpxlsf[mask_valid_corrf], confidencef[mask_valid_corrf]

                sample_num = 10000
                # Set replace True to prevent insufficient points
                np.random.seed(int(torch.sum(confidencef).int().item() * 100))
                confidencef = confidencef.cpu().numpy()
                good_samples = np.random.choice(
                    np.arange(len(correspondencef)),
                    size=sample_num,
                    replace=False,
                    p=confidencef / np.sum(confidencef),
                )

                # Check
                pts1 = kornia.geometry.conversions.convert_points_to_homogeneous(gridpxlsf[good_samples, :])
                pts2 = kornia.geometry.conversions.convert_points_to_homogeneous(correspondencef[good_samples, :])
                pts1norm, pts2norm = (intrinsic.inverse() @ pts1.T).T, (intrinsic.inverse() @ pts2.T).T
                pts1norm = kornia.geometry.conversions.convert_points_from_homogeneous(pts1norm)
                pts2norm = kornia.geometry.conversions.convert_points_from_homogeneous(pts2norm)

                if rel_norm_threshold_adjuster is not None:
                    flowmag = pts1norm - pts2norm
                    flowmag = torch.sqrt(torch.sum(flowmag ** 2, dim=-1))
                    flowmag_median = torch.median(flowmag).item()
                    norm_threshold = flowmag_median / 1000 * rel_norm_threshold_adjuster

                if use_opencv:
                    # Use OpenCV for Scale Estimation
                    pts1norm, pts2norm = pts1norm.cpu().numpy(), pts2norm.cpu().numpy()
                    conf = 0.99999
                    E, mask = cv2.findEssentialMat(
                        pts1norm, pts2norm, np.eye(3), threshold=norm_threshold, prob=conf, method=cv2.RANSAC
                    )
                    n, R, t, inliers = cv2.recoverPose(E, pts1norm, pts2norm, np.eye(3), 1e9, mask=mask)
                else:
                    from third_party.LightedDepth.GPUEPMatrixEstimation import gpuepm_function
                    R, t, inliers = gpuepm_function(
                        pts1norm, pts2norm, ransac_iter=5, ransac_threshold=norm_threshold
                    )
                est_rel_npose = np.eye(4)
                est_rel_npose[0:3, 0:3] = R
                est_rel_npose[0:3, 3:4] = t
                est_rel_npose = torch.from_numpy(est_rel_npose).float().cuda()

                if use_gtscale:
                    est_rel_pose = torch.clone(est_rel_npose)
                    est_rel_pose[0:3, 3] = est_rel_pose[0:3, 3] * (gt_rel_pose[0:3, 3] ** 2 + 1e-10).sum().sqrt()
                else:
                    from third_party.LightedDepth.utils.pose_estimation import npose2pose
                    pts1np, pts2np = kornia.geometry.conversions.convert_points_from_homogeneous(pts1), kornia.geometry.conversions.convert_points_from_homogeneous(pts2)
                    pts1np, pts2np, intrinsicnp = pts1np.cpu().numpy(), pts2np.cpu().numpy(), intrinsic.cpu().numpy()
                    mdn = depthmap[0, 0, pts1np[:, 1].astype(np.int_), pts1np[:, 0].astype(np.int_)].cpu().numpy()
                    est_rel_pose = npose2pose(pts1np, pts2np, mdn, intrinsicnp, est_rel_npose.cpu().numpy(), inliers)
                    est_rel_pose = torch.from_numpy(est_rel_pose).float().cuda()
                return est_rel_pose

            from source.utils.camera import pad_poses, unpad_poses
            images, intr, pose = self.train_data.all['image'], self.train_data.all['intr'], self.train_data.all['pose']

            n_views = images.shape[0]
            key_combi_list = generate_keypair_list(n_views)

            # Update for GT Correspondence
            corres_maps, conf_maps = self.flow_net.compute_matches_gtprojection(
                images, combi_list_tar_src=key_combi_list, plot=False, use_homography=False, additional_data=self.train_data.all
            )
            self.train_data.all['corres_gts_root2others'] = corres_maps

            # Update for Est Correspondence
            corres_maps, conf_maps = self.flow_net.compute_flow_and_confidence_map_of_combi_list(
                images, combi_list_tar_src=key_combi_list, plot=False, use_homography=False, additional_data=self.train_data.all
            )
            self.train_data.all['corres_est_root2others'] = corres_maps

            from source.training.core.correspondence_utils import get_mask_valid_from_conf_map
            _, _, H, W = corres_maps.shape
            mask_valid_corr = get_mask_valid_from_conf_map(
                p_r=conf_maps.reshape(-1, 1, H, W),
                corres_map=corres_maps.reshape(-1, 2, H, W),
                min_confidence=opt.min_conf_valid_corr)  # (n_views*(n_views-1), 1, H, W)

            if hasattr(self, 'monodepth_net'):
                with torch.no_grad():
                    if opt.mondepth_init == 'mscale_wtgt':
                        monodepths = self.monodepth_net.infer(images).squeeze(1)
                        for kk in range(len(monodepths)):
                            valid_depth_gt = self.train_data.all['valid_depth_gt'][kk].flatten()
                            depth_gt = self.train_data.all['depth_gt'][kk].flatten()
                            monodepth = monodepths[kk].flatten()
                            ratio = torch.median(depth_gt[valid_depth_gt]).item() / torch.median(monodepth[valid_depth_gt]).item()
                            monodepths[kk] = monodepths[kk] * ratio
                            assert (monodepths[kk].max() > self.train_data.all['depth_range'][kk, 0]) and (monodepths[kk].max() < self.train_data.all['depth_range'][kk, 1])
                    elif opt.mondepth_init == 'monodepth':
                        padder = InputPadder(dims=images.shape, padding=32)
                        images_padded, = padder.pad(images)
                        monodepths = list()
                        for didx in range(len(images_padded)):
                            if opt.mondepth_backbone == "ZeroDepth":
                                monodepths.append(self.monodepth_net(images_padded[didx:didx+1], intr[didx:didx+1]))
                            elif opt.mondepth_backbone == "ZoeDepth":
                                monodepths.append(self.monodepth_net.infer(images_padded[didx:didx+1]))
                            elif opt.mondepth_backbone == "Metric3DDepth":
                                image_PIL = images_padded[didx:didx+1].squeeze().permute([1, 2, 0]).cpu().numpy()
                                image_PIL = (image_PIL * 255.0).astype(np.uint8)
                                tmp_path = "{}.png".format(str(os.getpid()))
                                Image.fromarray(image_PIL).save(tmp_path)
                                image_cv2 = cv2.imread(tmp_path)
                                fx, fy, u0, v0 = intr[didx, 0, 0], intr[didx, 1, 1], intr[didx, 0, 2], intr[didx, 1, 2]
                                fx, fy, u0, v0 = fx.item(), fy.item(), u0.item(), v0.item()
                                intr_metric3d = np.array([fx, fy, u0, v0])
                                monodepths.append(self.monodepth_net.infer(image_cv2, intr_metric3d))
                                os.remove(tmp_path)
                            else:
                                raise NotImplementedError()
                        monodepths = torch.cat(monodepths, dim=0).squeeze(1)
                        monodepths = padder.unpad(monodepths)
                        # tmp_path = "{}.png".format(str(os.getpid()))
                        # from analysis.utils_vls import tensor2disp
                        # tensor2disp(1 / monodepths.unsqueeze(1), viewind=0, percentile=95).save(tmp_path)

                        # Median Scale the Root
                        kk = 0
                        valid_depth_gt = self.train_data.all['valid_depth_gt'][kk].flatten()
                        depth_gt = self.train_data.all['depth_gt'][kk].flatten()
                        monodepth = monodepths[kk].flatten()
                        ratio = torch.median(depth_gt[valid_depth_gt]).item() / torch.median(monodepth[valid_depth_gt]).item()
                        monodepths = monodepths * ratio

                    elif opt.mondepth_init == 'gtdepth':
                        monodepths = copy.deepcopy(self.train_data.all['depth_gt'])
                    else:
                        raise NotImplementedError()
                self.train_data.all.update({'depth_est': monodepths})

                min_depth, max_depth = monodepths[monodepths > 0].min().item(), monodepths[monodepths > 0].max().item()
                self.train_data.all['depth_range'][:, 0] = torch.ones_like(self.train_data.all['depth_range'][:, 0]) * min_depth * (1 - opt.increase_depth_range_by_x_percent)
                self.train_data.all['depth_range'][:, 1] = torch.ones_like(self.train_data.all['depth_range'][:, 1]) * max_depth * (1 + opt.increase_depth_range_by_x_percent)
            else:
                raise NotImplementedError()

            if 'gtdepth' in opt.camera.initial_pose:
                depthmap = self.train_data.all['depth_gt']
            elif 'estdepth' in opt.camera.initial_pose:
                depthmap = self.train_data.all['depth_est']
            elif 'gtscale' in opt.camera.initial_pose:
                depthmap = None
            else:
                raise ValueError()

            initial_poses_w2c = torch.eye(4, 4)[None, ...].repeat(n_poses, 1, 1).to(self.device)
            for idx in range(key_combi_list.shape[1]):
                target, source = key_combi_list[0, idx].item(), key_combi_list[1, idx].item()
                pose_target, pose_source = pad_poses(self.train_data.all['pose'][target]), pad_poses(self.train_data.all['pose'][source])
                gt_rel_pose = unpad_poses(pose_source @ pose_target.inverse())

                use_opencv = 'opencv' in opt.camera.initial_pose
                use_gtscale = 'gtscale' in opt.camera.initial_pose

                if depthmap is not None:
                    depthmap_indexed = depthmap[idx].unsqueeze(0).unsqueeze(0)
                else:
                    depthmap_indexed = None

                est_rel_pose = twoview_pose_estimation(
                    corres_maps[idx].unsqueeze(0),
                    conf_maps[idx].unsqueeze(0),
                    mask_valid_corr[idx].unsqueeze(0),
                    depthmap=depthmap_indexed,
                    intrinsic=intr[0],
                    use_opencv=use_opencv,
                    use_gtscale=use_gtscale,
                    minconf=opt.min_conf_valid_corr,
                    gt_rel_pose=gt_rel_pose,
                    norm_threshold_adjuster=opt.norm_threshold_adjuster,
                    rel_norm_threshold_adjuster=opt.rel_norm_threshold_adjuster
                )

                initial_poses_w2c[idx + 1] = initial_poses_w2c[0] @ est_rel_pose
        else:
                raise ValueError
        return initial_poses_w2c[:, :3], valid_poses_idx, index_images_excluded

    @torch.no_grad()
    def prealign_w2c_large_camera_systems(self,opt: Dict[str, Any],pose_w2c: torch.Tensor,
                                          pose_GT_w2c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the 3D similarity transform relating pose_w2c to pose_GT_w2c. Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)

        if self.settings.camera.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=self.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            try:
                pose_aligned_c2w, ssim_est_gt_c2w = align_ate_c2b_use_a2b(pose_c2w, pose_GT_c2w, method='sim3')
                pose_aligned_w2c = camera.pose.invert(pose_aligned_c2w[:, :3])
                ssim_est_gt_c2w.type = 'traj_align'
            except:
                self.logger.info("warning: SVD did not converge...")
                pose_aligned_w2c = pose_w2c
                ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), type='traj_align', 
                                        t=torch.zeros(1,3,1,device=self.device), s=1.)
        return pose_aligned_w2c, ssim_est_gt_c2w

    @torch.no_grad()
    def prealign_w2c_small_camera_systems(self,opt: Dict[str, Any],pose_w2c: torch.Tensor,
                                          pose_GT_w2c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the transformation from pose_w2c to pose_GT_w2c by aligning the each pair of pose_w2c 
        to the corresponding pair of pose_GT_w2c and computing the scaling. This is more robust than the
        technique above for small number of input views/poses (<10). Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        def alignment_function(poses_c2w_from_padded: torch.Tensor, 
                               poses_c2w_to_padded: torch.Tensor, idx_a: int, idx_b: int):
            """Args: FInd alignment function between two poses at indixes ix_a and idx_n

                poses_c2w_from_padded: Shape is (B, 4, 4)
                poses_c2w_to_padded: Shape is (B, 4, 4)
                idx_a:
                idx_b:

            Returns:
            """
            # We take a copy to keep the original poses unchanged.
            poses_c2w_from_padded = poses_c2w_from_padded.clone()
            # We use the distance between the same two poses in both set to obtain
            # scale misalgnment.
            dist_from = torch.norm(
                poses_c2w_from_padded[idx_a, :3, 3] - poses_c2w_from_padded[idx_b, :3, 3]
            )
            dist_to = torch.norm(
                poses_c2w_to_padded[idx_a, :3, 3] - poses_c2w_to_padded[idx_b, :3, 3])
            scale = dist_to / dist_from

            # alternative for scale
            # dist_from = poses_w2c_from_padded[idx_a, :3, 3] @ poses_c2w_from_padded[idx_b, :3, 3]
            # dist_to = poses_w2c_to_padded[idx_a, :3, 3] @ poses_c2w_to_padded[idx_b, :3, 3]
            # scale = onp.abs(dist_to /dist_from).mean()

            # We bring the first set of poses in the same scale as the second set.
            poses_c2w_from_padded[:, :3, 3] = poses_c2w_from_padded[:, :3, 3] * scale

            # Now we simply apply the transformation that aligns the first pose of the
            # first set with first pose of the second set.
            transformation_from_to = poses_c2w_to_padded[idx_a] @ camera.pose_inverse_4x4(
                poses_c2w_from_padded[idx_a])
            poses_aligned_c2w = transformation_from_to[None] @ poses_c2w_from_padded

            poses_aligned_w2c = camera.pose_inverse_4x4(poses_aligned_c2w)
            ssim_est_gt_c2w = edict(R=transformation_from_to[:3, :3].unsqueeze(0), type='traj_align', 
                                    t=transformation_from_to[:3, 3].reshape(1, 3, 1), s=scale)

            return poses_aligned_w2c[:, :3], ssim_est_gt_c2w

        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
        B = pose_c2w.shape[0]

        if self.settings.camera.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=self.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            # try every combination of pairs and get the rotation/translation
            # take the one with the smallest error
            # this is because for small number of views, the procrustes alignement with SVD is not robust. 
            pose_aligned_w2c_list = []
            ssim_est_gt_c2w_list = []
            error_R_list = []
            error_t_list = []
            full_error = []
            for pair_id_0 in range(min(B, 10)):  # to avoid that it is too long
                for pair_id_1 in range(min(B, 10)):
                    if pair_id_0 == pair_id_1:
                        continue
                    
                    pose_aligned_w2c_, ssim_est_gt_c2w_ = alignment_function\
                        (camera.pad_poses(pose_c2w), camera.pad_poses(pose_GT_c2w),
                         pair_id_0, pair_id_1)
                    pose_aligned_w2c_list.append(pose_aligned_w2c_)
                    ssim_est_gt_c2w_list.append(ssim_est_gt_c2w_ )

                    error = self.evaluate_camera_alignment(opt, pose_aligned_w2c_, pose_GT_w2c)
                    error_R_list.append(error.R.mean().item() * 180. / np.pi )
                    error_t_list.append(error.t.mean().item())
                    full_error.append(error.t.mean().item() * (error.R.mean().item() * 180. / np.pi))

            ind_best = np.argmin(full_error)
            # print(np.argmin(error_R_list), np.argmin(error_t_list), ind_best)
            pose_aligned_w2c = pose_aligned_w2c_list[ind_best]
            ssim_est_gt_c2w = ssim_est_gt_c2w_list[ind_best]

        return pose_aligned_w2c, ssim_est_gt_c2w

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt: Dict[str, Any],pose_aligned_w2c: torch.Tensor,
                                  pose_GT_w2c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Measures rotation and translation error between aligned and ground-truth world-to-camera poses. 
        Attention, we want the translation difference between the camera centers in the world 
        coordinate! (not the opposite!)
        Args:
            opt (edict): settings
            pose_aligned_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        Returns:
            error: edict with keys 'R' and 't', for rotation (in radian) and translation erorrs (not averaged)
        """
        # just invert both poses to camera to world
        # so that the translation corresponds to the position of the camera in world coordinate frame. 
        pose_aligned_c2w = camera.pose.invert(pose_aligned_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)

        R_aligned_c2w,t_aligned_c2w = pose_aligned_c2w.split([3,1],dim=-1)
        # R_aligned is (B, 3, 3)
        t_aligned_c2w = t_aligned_c2w.reshape(-1, 3)  # (B, 3)

        R_GT_c2w,t_GT_c2w = pose_GT_c2w.split([3,1],dim=-1)
        t_GT_c2w = t_GT_c2w.reshape(-1, 3)

        R_error = camera.rotation_distance(R_aligned_c2w,R_GT_c2w)
        
        t_error = (t_aligned_c2w - t_GT_c2w).norm(dim=-1)

        error = edict(R=R_error,t=t_error)  # not meaned here
        return error
    
    @torch.no_grad()
    def evaluate_any_poses(self, opt: Dict[str, Any], pose_w2c: torch.Tensor, 
                           pose_GT_w2c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluates rotation and translation errors before and after alignment. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        stats_dict = {}
        if pose_w2c.shape[0] > 10:
            pose_aligned,_ = self.prealign_w2c_large_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        else:
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        error = self.evaluate_camera_alignment(opt,pose_aligned, pose_GT_w2c)
        stats_dict['error_t'] = error.t.mean().item()
        stats_dict.update(self.evaluate_camera_relative(pose_w2c, pose_GT_w2c))
        return stats_dict

    @torch.no_grad()
    def evaluate_camera_relative(self, pose_w2c, pose_GT_w2c):
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

        error_ts, error_Rs = list(), list()
        for i in range(1, len(pose_w2c)):
            error_t, error_R = compute_relpose_error_deg(relpose_GT[i], relpose[i])
            error_ts.append(error_t), error_Rs.append(error_R)
        error_t, error_R = np.array(error_ts).mean(), np.array(error_Rs).mean()

        stats_dict = {
            'error_R_rel_deg': error_R,
            'error_t_rel_deg': error_t
        }
        return stats_dict

    @torch.no_grad()
    def evaluate_poses(self, opt: Dict[str, Any], 
                       idx_optimized_pose: List[int]=None) -> Dict[str, torch.Tensor]: 
        pose,pose_GT = self.get_all_training_poses(opt, idx_optimized_pose=idx_optimized_pose)
        return self.evaluate_any_poses(opt, pose, pose_GT)
        # return self.evaluate_camera_relative(pose, pose_GT)

    @torch.no_grad()
    def evaluate_keydepth(self, opt: Dict[str, Any], data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.net.eval()
        img_idx, iteration = 0, -1
        output_dict = self.net.render_image_at_specific_rays \
            (self.settings, data_dict, img_idx=img_idx, iter=iteration, mode="train")
        _, _, H, W = data_dict['image'].shape
        output_dict['depth'] = output_dict['depth'].view([1, H, W])
        self.net.train()

        for x in self.loss_module.loss_modules:
            if isinstance(x, CorrespondencesPairRenderDepthAndGet3DPtsAndReproject) or isinstance(x, TriangulationLoss):
                break
        assert isinstance(x, CorrespondencesPairRenderDepthAndGet3DPtsAndReproject) or isinstance(x, TriangulationLoss)
        accum_corr_corect_loc = x.acq_accum_corr_corect_loc(img_idx)
        accum_corr_corect_loc = accum_corr_corect_loc >= self.settings.train_sub - 2

        def compute_depth_error_scaled(data_dict, output_dict, pred_depth, external_mask=None):
            return compute_depth_error(
                data_dict=data_dict,
                output_dict=output_dict,
                pred_depth=pred_depth,
                scaling_factor_for_pred_depth=-1.0,
                external_mask=external_mask
            )

        abs_pred, rmse_pred, a1_pred = compute_depth_error_scaled(
            data_dict, output_dict,
            pred_depth=output_dict['depth'], external_mask=accum_corr_corect_loc
        )
        abs_init, rmse_init, a1_init = compute_depth_error_scaled(
            data_dict, output_dict,
            pred_depth=data_dict['depth_est'][output_dict['idx_img_rendered'][0].item()], external_mask=accum_corr_corect_loc
        )
        abs_pred_improve, rmse_pred_improve, a1_pred_improve = abs_init - abs_pred, rmse_init - rmse_pred, a1_pred - a1_init

        return {'abs_pred': abs_pred,
                'rmse_pred': rmse_pred,
                'a1_pred': a1_pred,
                'abs_pred_improve': abs_pred_improve,
                'rmse_pred_improve': rmse_pred_improve,
                'a1_pred_improve': a1_pred_improve
                }

    def visualize_any_poses(self, opt: Dict[str, Any], pose_w2c: torch.Tensor, pose_ref_w2c: torch.Tensor, 
                            step: int=0, idx_optimized_pose: List[int]=None, split: str="train") -> Dict[str, torch.Tensor]:
        """Plot the current poses versus the reference ones, before and after alignment. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_ref_w2c (torch.Tensor): Shape is (B, 3, 4)
            step (int, optional): Defaults to 0.
            split (str, optional): Defaults to "train".
        """
        
        plotting_dict = {}
        fig = plt.figure(figsize=(20,10) if 'nerf_synthetic' in opt.dataset else ((16,8)))

        # we show the poses without the alignment here
        pose_aligned, pose_ref = pose_w2c.detach().cpu(), pose_ref_w2c.detach().cpu()
        if 'llff' in opt.dataset:
            pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,ep=self.iteration)
        else:
            pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref, ep=self.iteration)
        
        '''
        if self.settings.debug:
            import imageio
            imageio.imwrite('pose_before_align_{}.png'.format(self.iteration), pose_vis)
        '''
        pose_vis = torch.from_numpy(pose_vis.astype(np.float32)/255.).permute(2, 0, 1)
        plotting_dict['poses_before_align'] = pose_vis

        # trajectory alignment
        if pose_w2c.shape[0] > 9:
            # the alignement will work well when more than 10 poses are available
            pose_aligned,_ = self.prealign_w2c_large_camera_systems(opt,pose_w2c.detach(),pose_ref_w2c)
        else:
            # for few number of images/poses, the above alignement doesnt work well
            # align the first camera and scale with the relative to the second
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose_w2c.detach(),pose_ref_w2c)
        
        pose_aligned = pose_aligned.detach().cpu()
        if 'llff' in opt.dataset:
            pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                ep=self.iteration)
        else:
            pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                        ep=self.iteration)
        

        pose_vis = torch.from_numpy(pose_vis.astype(np.float32)/255.).permute(2, 0, 1)
        plotting_dict['poses_after_align'] = pose_vis
        return plotting_dict


    def visualize_poses(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                        step: int=0,idx_optimized_pose: List[int]=None, 
                        split: str="train") -> Dict[str, torch.Tensor]:
        pose, pose_ref_ = self.get_all_training_poses(opt, idx_optimized_pose=idx_optimized_pose)
        return self.visualize_any_poses(opt, pose, pose_ref_, step, split=split)
    
    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self, opt: Dict[str, Any], 
                                             data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run test-time optimization. Optimizes over data_dict.se3_refine_test"""
        # only optimizes for the test pose here
        data_dict.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=self.device))
        optimizer = getattr(torch.optim,opt.optim.algo_pose)
        optim_pose = optimizer([dict(params=[data_dict.se3_refine_test],lr=opt.optim.lr_pose)])
        #iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in range(opt.optim.test_iter):
            optim_pose.zero_grad()
            
            data_dict.pose_refine_test = camera.lie.se3_to_SE3(data_dict.se3_refine_test)
            output_dict = self.net.forward(opt,data_dict,mode="test-optim", iter=None)

            # current estimate of the pose
            poses_w2c = self.net.get_pose(self.settings, data_dict, mode='test-optim')  # is it world to camera
            data_dict.poses_w2c = poses_w2c

            # iteration needs to reflect the overall training
            loss, stats_dict, plotting_dict = self.loss_module.compute_loss\
                (opt, data_dict, output_dict, iteration=self.iteration, mode='test-optim')
            loss.all.backward()
            optim_pose.step()
            # iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return data_dict



class PoseAndNerfTrainerPerScene(nerf.NerfTrainerPerScene, CommonPoseEvaluation):
    """Base class for joint pose-NeRF training. Inherits from NeRFTrainerPerScene. 
    """
    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)
        self.training_start_time = time.time()

    def build_pose_net(self, opt: Dict[str, Any]):
        """Defines initial poses. Define parametrization of the poses. 
        """
        
        # get initial poses (which will be optimized)
        # if load_colmap_depth, it will be integrated to the data here! 
        initial_poses_w2c, valid_poses_idx, index_images_excluded = self.set_initial_poses(opt)

        assert not opt.load_colmap_depth
        assert len(valid_poses_idx) + len(index_images_excluded) == initial_poses_w2c.shape[0]

        # log and save to tensorboard initial pose errors
        self.logger.critical('Found {}/{} valid initial poses'\
            .format(len(valid_poses_idx), initial_poses_w2c.shape[0]))

        # evaluate initial poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(self.device)

        stats_dict = self.evaluate_any_poses(opt, initial_poses_w2c, pose_GT)
        self.initial_pose_error = stats_dict

        message = get_log_string(stats_dict)
        self.logger.critical('All initial poses: {}'.format(message))
        self.write_event('train', {'nbr_excluded_poses': len(index_images_excluded)}, self.iteration)
        self.write_event('train', stats_dict, self.iteration)

        assert opt.camera.pose_parametrization != 'two_columns'
        # define pose parametrization
        self.pose_net = FirstTwoColunmnsScalePoseOptDepthParameters(
            opt, nbr_poses=len(self.train_data), initial_poses_w2c=initial_poses_w2c, device=self.device
        )
        self.export_pose_estimate_for_evaluation()
        return

    def export_pose_estimate_for_evaluation(self):
        self.logger.info("Exporting the Initial Poses.........")
        import pickle
        train_data = copy.deepcopy(self.train_data.all)
        for key in train_data.keys():
            if isinstance(train_data[key], torch.Tensor):
                train_data[key] = train_data[key].cpu()
        poses_w2c = self.pose_net.get_w2c_poses().detach().cpu()

        data_input = os.path.join(self.writer.log_dir, 'input_data.pickle')
        file = open(data_input, 'wb')
        pickle.dump(train_data, file)
        file.close()

        init_pose_input = os.path.join(self.writer.log_dir, 'init_pose.pickle')
        file = open(init_pose_input, 'wb')
        pickle.dump(poses_w2c, file)
        file.close()

    def export_pose_params_for_evaluation(self, tag):
        param_path = os.path.join(self.writer.log_dir, '{}.pth'.format(tag))
        self.logger.info("Exporting the Optimized Poses %s to %s" % (tag, param_path))
        with open(param_path, 'wb') as f:
            torch.save(self.pose_net.state_dict(), f)

    def export_corres_from_pose_for_evaluation(self):
        corr_path = os.path.join(self.writer.log_dir, 'correspondence.pth')
        self.logger.info("Exporting the Correspondence Estimate to %s" % corr_path)
        with open(corr_path, 'wb') as f:
            torch.save(self.pose_net.corres_estimate_bundle, f)

    def build_nerf_net(self, opt: Dict[str, Any], pose_net: torch.nn.Module):
        self.logger.info('Creating NerF model for joint pose-NeRF training')
        self.net = Graph(opt, self.device, pose_net)
        return

    def build_networks(self,opt: Dict[str, Any]):
        self.logger.info("building networks...")

        if opt.use_flow:
            self.build_correspondence_net(opt)

        if opt.use_monodepth:
            self.build_monodepth_net(opt)

        self.build_pose_net(opt) 

        self.build_nerf_net(opt, self.pose_net)

        return 

    def setup_optimizer(self,opt: Dict[str, Any]):
        super().setup_optimizer(opt)
        self.logger.info('setting up optimizer for camera poses')
        optimizer = getattr(torch.optim,opt.optim.algo_pose)
        self.optimizer_pose = optimizer([dict(
            params=self.pose_net.parameters(),
            lr=opt.optim.lr_pose)])
        # set up scheduler
        if opt.optim.sched_pose:
            self.logger.info('setting up scheduler for camera poses')
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                max_iter = opt.optim.max_iter if hasattr(opt.optim, 'max_iter') else opt.max_iter
                if opt.optim.lr_pose > 0.0:
                    opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./max_iter)
                else:
                    opt.optim.sched_pose.gamma = 0.0
            else:
                assert opt.optim.lr_pose_end == 0.0 and opt.optim.lr_pose == 0.0
                opt.optim.sched_pose.gamma = 1.0

            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            self.scheduler_pose = scheduler(self.optimizer_pose,**kwargs)
        return 
        
    def update_parameters(self, loss_out: Dict[str, Any]):
        """Update NeRF MLP parameters and pose parameters"""
        if self.settings.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optimizer_pose.param_groups[0]["lr_orig"] = self.optimizer_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optimizer_pose.param_groups[0]["lr"] *= min(1,self.iteration/self.settings.optim.warmup_pose)

        loss_out.backward()

        # camera update
        if self.iteration % self.grad_acc_steps == 0:
            do_backprop = self.after_backward(self.pose_net, self.iteration, 
                                              gradient_clipping=self.settings.pose_gradient_clipping)

            if self.iteration < self.settings.start_iter.pose:
                # Avoid Finetune Pose
                self.optimizer_pose.zero_grad()

            if do_backprop:
                self.optimizer_pose.step()

            self.optimizer_pose.zero_grad()  # puts the gradient to zero for the poses

            if self.settings.optim.warmup_pose:
                self.optimizer_pose.param_groups[0]["lr"] = self.optimizer_pose.param_groups[0]["lr_orig"] 
                # reset learning rate
            if self.settings.optim.sched_pose: self.scheduler_pose.step()

        # backprop of the nerf network
        do_backprop = self.after_backward(self.net.get_network_components(), self.iteration, 
                                          gradient_clipping=self.settings.nerf_gradient_clipping)
        if self.iteration % self.grad_acc_steps == 0:
            if do_backprop:
                # could be skipped in case of large gradients or something, but we still want to put the optimizer
                # gradient to zero, so it is not further accumulated
                # skipped the accumulation gradient step
                self.optimizer.step()
            self.optimizer.zero_grad()

            # scheduler, here after each step
            if self.scheduler is not None: self.scheduler.step()
        return

    @torch.no_grad()
    def inference(self):
        pose, pose_GT = self.get_all_training_poses(self.settings)

        # computes the alignement between optimized poses and gt ones. Will be used to transform the test
        # poses to coordinate system of optimized poses for the evaluation. 
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().inference()
        return 

    @torch.no_grad()
    def inference_debug(self):
        pose,pose_GT = self.get_all_training_poses(self.settings)
        # computes the alignement between optimized poses and gt ones. Will be used to transform the test
        # poses to coordinate system of optimized poses for the evaluation. 
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().inference_debug()
        return 

    @torch.no_grad()
    def generate_videos_synthesis(self, opt: Dict[str, Any]):
        pose,pose_GT = self.get_all_training_poses(self.settings)
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().generate_videos_synthesis(opt)
        return 

    @torch.no_grad()
    def make_result_dict(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                         loss: Dict[str, Any],metric: Dict[str, Any]=None,split: str="train"):
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = super().make_result_dict(opt,data_dict,output_dict,loss,metric=metric,split=split)
        if split=="train":
            # log learning rate
            if hasattr(self, 'optimizer_pose'):
                lr = self.optimizer_pose.param_groups[0]["lr"]
                stats_dict["lr_pose"] = lr
        return stats_dict

    @torch.no_grad()
    def make_results_dict_low_freq(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                                   loss: Dict[str, Any],metric: Dict[str, Any]=None,split:str="train") -> Dict[str, torch.Tensor]:
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = {}
        # compute pose error, this is heavy, so just compute it after x iterations
        if split == "train":
            stats_dict.update(self.evaluate_poses(opt))
            # stats_dict.update(self.evaluate_keydepth(opt, data_dict))
        return stats_dict

    @torch.no_grad()
    def visualize(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                  step: int=0, split: str="train") -> Dict[str, torch.Tensor]:
        plotting_dict = super().visualize(opt,data_dict, output_dict, step=step,split=split)

        '''
        # Temporarily Removed
        if split == 'train' and step == 0:
            # only does that once
            plotting_dict_ = self.visualize_poses(opt,data_dict,output_dict,step,split=split)
            plotting_dict.update(plotting_dict_)
        '''
        return plotting_dict

    @torch.no_grad()
    def get_all_training_poses(self,opt: Dict[str, Any], 
                               idx_optimized_pose: List[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(self.device)
        pose_w2c = self.net.pose_net.get_w2c_poses()  
        if idx_optimized_pose is not None:
            pose_GT_w2c = pose_GT_w2c[idx_optimized_pose].reshape(-1, 3, 4)
            pose_w2c = pose_w2c[idx_optimized_pose].reshape(-1, 3, 4)
        return pose_w2c, pose_GT_w2c

    @torch.no_grad()
    def evaluate_full(self,opt: Dict[str, Any], plot: bool=False, save_ind_files: bool=False, 
                      out_scene_dir: str='') -> Dict[str, torch.Tensor]:
        self.net.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)
        if pose.shape[0] > 9:
            # alignment of the trajectory
            pose_aligned, self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(opt,pose,pose_GT)
        else:
            # alignment of the first cameras
            pose_aligned, self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(opt,pose,pose_GT)
            
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        self.logger.info("--------------------------")
        self.logger.info("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        self.logger.info("trans: {:10.5f}".format(error.t.mean()))
        self.logger.info("--------------------------")
        # dump numbers
        # evaluate novel view synthesis
        results_dict = super().evaluate_full(opt, plot=plot, out_scene_dir=out_scene_dir)
        results_dict['rot_error'] = np.rad2deg(error.R.mean().item())
        results_dict['trans_error'] = error.t.mean().item()

        # init error
        results_dict['init_rot_error']= self.initial_pose_error['error_R_before_align'].item()
        results_dict['init_trans_error'] = self.initial_pose_error['error_t_before_align'].item()
        return results_dict

    @torch.no_grad()
    def generate_videos_pose(self,opt: Dict[str, Any]):
        self.net.eval()

        opt.output_path = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []

        for ep in range(0,opt.max_iter+1,self.snapshot_steps):
            # load checkpoint (0 is random init)
            if ep!=0:
                checkpoint_path = '{}/{}/iter-{:04d}.pth.tar'.format(self._base_save_dir, self.settings.project_path, ep)
                if not os.path.exists(checkpoint_path):
                    continue
                self.load_snapshot(checkpoint=ep)

            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose.detach(),pose_ref)
            # self.evaluate_camera_alignment(opt,pose_aligned,pose_ref_)
            pose_aligned = pose_aligned.detach().cpu()
            pose_ref = pose_ref.detach().cpu()

            fig = plt.figure(figsize=(16,8))
            if 'llff' in opt.dataset:
                pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                    ep=ep, path=cam_path)
            else:
                pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                            ep=ep, path=cam_path)
            ep_list.append(ep)
            plt.close()
        # write videos
        self.logger.info("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 20 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)
        return 
        
        
# ============================ computation graph for forward/backprop ============================

class Graph(Graph):

    def __init__(self, opt: Dict[str, Any], device: torch.device, pose_net: torch.nn.Module):
        super().__init__(opt, device)

        # nerf networks already done 
        self.pose_net = pose_net

    def get_w2c_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],
                     mode: str=None) -> torch.Tensor:
        if mode=="train":
            pose = self.pose_net.get_w2c_poses()  # get the current estimates of the camera poses, which are optimized
        elif mode in ["val","eval","test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end 
            # align test pose to refined coordinate system (up to sim3)
            if hasattr(self, 'sim3_est_to_gt_c2w'):
                pose_GT_w2c = data_dict.pose
                ssim_est_gt_c2w = self.sim3_est_to_gt_c2w
                if ssim_est_gt_c2w.type == 'align_to_first':
                    pose = backtrack_from_aligning_and_scaling_to_first_cam(pose_GT_w2c, ssim_est_gt_c2w)
                elif ssim_est_gt_c2w.type == 'traj_align':
                    pose = backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w)
                else:
                    raise ValueError
                # Here, we align the test pose to the poses found during the optimization (otherwise wont be valid)
                # that's pose. And can learn an extra alignement on top
                # additionally factorize the remaining pose imperfection
                if opt.optim.test_photo and mode!="val":
                    pose = camera.pose.compose([data_dict.pose_refine_test,pose])
            else:
                pose = self.pose_net.get_w2c_poses()
        else: 
            raise ValueError
        return pose

    def get_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        return self.get_w2c_pose(opt, data_dict, mode)

    def get_c2w_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        w2c = self.get_w2c_pose(opt, data_dict, mode)
        return camera.pose.invert(w2c)
