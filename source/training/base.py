
import matplotlib.pyplot as plt
import numpy as np
import os,sys,time
import torch
import torch.utils.tensorboard
import json
from easydict import EasyDict as edict
import lpips
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
import PIL.Image as Image

from third_party.pytorch_ssim.ssim import ssim as ssim_loss
from source.datasets.create_dataset import create_dataset
from source.training.engine.iter_based_trainer import IterBasedTrainer
from source.training.engine.iter_based_trainer import get_log_string
from source.training.core.loss_factory import define_loss

from source.training.core.triangulation_loss import TriangulationLoss
from analysis.utils_vls import tensor2disp, tensor2rgb
# ============================ main engine for training and evaluation ============================
        
class PerSceneTrainer(IterBasedTrainer):
    """Base class for NeRF or joint pose-NeRF training and evaluation
    """

    def __init__(self,opt: Dict[str, Any]):
        super().__init__(settings=opt, max_iteration=opt.max_iter, snapshot_steps=opt.snapshot_steps, 
                         grad_acc_steps=opt.grad_acc_steps)

        self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)

    def define_loss_module(self, opt: Dict[str, Any]):
        self.logger.info(f'Defining the loss: {opt.loss_type}')
        flow_net = self.flow_net if opt.use_flow else None

        self.loss_module = define_loss(opt.loss_type, opt, self.net, self.train_data, 
                                        self.device, flow_net=flow_net)

        '''
        # Temporarily Removed
        # save the matching in tensorboard, when using flow_net
        to_plot = self.loss_module.plot_something()
        self.write_image('train', to_plot, self.iteration)
        '''

        # save the initial correspondence metrics, if ground-truth depth/poses are available
        epe_stats = self.loss_module.get_flow_metrics()
        self.write_event('train', epe_stats, self.iteration)
        return 

    def load_dataset(self,opt: Dict[str, Any],eval_split: str="val"):
        self.logger.info("loading training data...")
        self.train_data, train_sampler = create_dataset(opt, mode='train')
        train_loader = self.train_data.setup_loader(shuffle=True)
        
        self.logger.info("loading test data...")
        if opt.val_on_test: eval_split = "test"
        self.test_data = create_dataset(opt, mode=eval_split)
        test_loader = self.test_data.setup_loader(shuffle=False)
        self.register_loader(train_loader, test_loader)  # save them 
        return 

    def build_networks(self,opt: Dict[str, Any]):
        raise NotImplementedError

    def train_iteration_nerf(self, data_dict: Dict[str, Any]):
        """ Run one iteration of training. 
        Only the NeRF mlp is trained
        The pose network (if applicable) is frozen
        """
        # put to not training the weights of pose_net
        if hasattr(self, 'pose_net'):
            for p in self.pose_net.parameters():
                p.requires_grad = False
            self.pose_net.eval()

        # forward
        output_dict, result_dict, plotting_dict = self.train_step\
            (self.iteration, data_dict)

        # backward & optimization
        result_dict['loss'].backward()
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
        return output_dict, result_dict, plotting_dict

    def train_iteration_nerf_pose_flow(self, data_dict: Dict[str, Any]):
        """ Run one iteration of training
        The nerf mlp is optimized
        The poses are also potentially optimized
        """
        self.iteration_nerf += 1

        # forward
        output_dict, result_dict, plotting_dict = self.train_step(self.iteration, data_dict)
        
        # backward & optimization
        self.update_parameters(result_dict['loss'])
        return output_dict, result_dict, plotting_dict


    def update_parameters(self, loss_out: Dict[str, Any]):
        """ Update weights of mlp"""
        loss_out.backward()
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


    def train_iteration(self, data_dict: Dict[str, Any]):
        self.before_train_step(self.iteration, data_dict)
        self.timer.add_prepare_time()

        output_dict, result_dict, plotting_dict = self.train_iteration_nerf_pose_flow(data_dict)

        # after training
        self.timer.add_process_time()
        self.after_train_step(self.iteration, data_dict, output_dict, result_dict)
        result_dict = self.release_tensors(result_dict)
        
        self.summary_board.update_from_result_dict(result_dict)
        self.write_image('train', plotting_dict, self.iteration)
        return result_dict


    def set_train_mode(self):
        self.training = True
        self.net.train()
        if self.settings.use_flow and hasattr(self, 'flow_net'):
            # THIS IS IMPORTANT
            # should always be in eval mode
            self.flow_net.eval()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.net.eval()
        if self.settings.use_flow and hasattr(self, 'flow_net'):
            self.flow_net.eval()
        torch.set_grad_enabled(False)
    
    def return_model_dict(self):
        state_dict = {}
        state_dict['nerf_net'] = self.net.state_dict()

        if hasattr(self, 'pose_net'):
            state_dict['pose_net'] = self.pose_net.state_dict()
        return state_dict
    
    def load_state_dict(self, model_dict, do_partial_load=False):

        if not 'nerf_net' in model_dict.keys():
            if do_partial_load:
                partial_load(model_dict, self.net, skip_keys=[])
            else:
                self.net.load_state_dict(model_dict, strict=True)   
            return
        
        assert 'nerf_net' in model_dict.keys()
        self.logger.info('Loading the nerf model')
        if do_partial_load:
            partial_load(model_dict['nerf_net'], self.net, skip_keys=[])
        else:
            incompactible = self.net.load_state_dict(model_dict['nerf_net'], strict=False)
            self.logger.info(incompactible)

        if 'flow_net' in model_dict.keys():
            self.logger.info('Loading flow net')
            self.flow_net.load_state_dict(model_dict['flow_net'], strict=True)

        if 'pose_net' in model_dict.keys():
            self.logger.info('Loading the poses')
            self.pose_net.load_state_dict(model_dict['pose_net'], strict=True)
        return 

    def generate_videos_pose(self, opt: Dict[str, Any]):
        raise NotImplementedError

    @torch.no_grad()
    def visualize_nerf(self, load_latest: bool=True):
        if load_latest:
            self.load_snapshot()

        from source.utils.helper import acquire_depth_range
        depth_range = acquire_depth_range(self.settings, self.train_data.all)
        n_samples = self.settings.nerf.sample_intvs
        depth_min, depth_max = depth_range
        depth_samples = 0.5 * torch.ones(1,1,n_samples,1,device=self.device)
        depth_samples += torch.arange(n_samples,device=self.device)[None,None,:,None].float() # the added part is [1, 1, N, 1] ==> [B,HW,N,1]
        depth_samples = depth_samples/n_samples*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = (1 / depth_samples).squeeze()

        depth_gt = self.train_data.all['depth_gt']
        image = tensor2rgb(self.train_data.all['image'], viewind=0)
        rgbw, rgbh = image.size

        output_dict = self.net.render_image_at_specific_rays \
            (self.settings, self.train_data.all, iter=100000, mode="val", img_idx=[0], per_slice=False)
        density = 1 - output_dict['T']
        density = density.view([1, rgbh, rgbw, 128])
        ch = density.shape[3]

        depth = output_dict['depth'].view([1, 1, rgbh, rgbw])
        tensor2disp(1 / depth, vmax=1.0, viewind=0).show()

        depth_vls = list()
        for i in range(self.settings.train_sub):
            output_dict = self.net.render_image_at_specific_rays \
                (self.settings, self.train_data.all, iter=100000, mode="val", img_idx=[i], per_slice=False)
            fig1 = np.array(tensor2disp(1 / output_dict['depth'].view([1, 1, rgbh, rgbw]), vmax=1.0, viewind=0))
            fig2 = np.array(tensor2rgb(self.train_data.all['image'], viewind=i))
            depth_vls.append(np.concatenate([fig1, fig2], axis=1))
        depth_vls = np.concatenate(depth_vls, axis=0)
        Image.fromarray(depth_vls).show()

        # Analysis the bacjprojected points to the NeRF Volume
        # for i in range(self.settings.train_sub):


        for i in range(ch):
            occu_vls_path = os.path.join(self.writer.log_dir, 'nerf_core_layer_{}.jpg'.format(str(i).zfill(3)))
            fig1 = tensor2disp(density[0, :, :, i].view([1, 1, rgbh, rgbw]), vmax=1.0)

            if i == 0:
                minval, maxval = 0, depth_samples[i].item()
            else:
                minval, maxval = depth_samples[i-1].item(), depth_samples[i].item()

            selector = (depth_gt < maxval)
            fig2 = tensor2disp(selector.unsqueeze(1).float(), vmax=1.0)

            figcombined = np.concatenate(
                [np.array(fig1), np.array(fig2), np.array(image)], axis=1
            )
            Image.fromarray(figcombined).save(occu_vls_path)

    @torch.no_grad()
    def analysis(self, load_latest: bool=True):
        if load_latest:
            self.load_snapshot()

        # load Data
        data_dict = self.train_data.all
        data_dict['iter'] = self.iteration
        B, _, H, W = data_dict['image'].shape

        self.net.eval()

        # current estimate of the pose
        poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='train')  # is it world to camera
        data_dict.poses_w2c = poses_w2c

        """
        # Render Root Frame Depth
        img_idx = 0
        output_dict = self.net.render_image_at_specific_rays \
            (self.settings, data_dict, img_idx=img_idx, iter=0, mode="train")

        root_frame_depth = output_dict['depth'].view([1, 1, H, W])
        from analysis.utils_vls import tensor2disp
        tensor2disp(1 / root_frame_depth, vmax=1.0, viewind=0).show()
        """

        if self.settings.nerf.depth.param == 'inverse':
            depth_range = self.settings.nerf.depth.range
            depth_range_in_self, depth_range_in_other = depth_range, depth_range
        elif self.settings.nerf.depth.param == 'monoguided':
            depth_range = data_dict['depth_est']
            depth_range_in_self, depth_range_in_other = depth_range[id_self].unsqueeze(0), depth_range[id_matching_view].unsqueeze(0)
        elif self.settings.nerf.depth.param == 'datainverse':
            depth_range = [1 / data_dict.depth_range[0, 0].item(), 1 / data_dict.depth_range[0, 1].item()]
            depth_range_in_self, depth_range_in_other = depth_range, depth_range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]
            depth_range_in_self, depth_range_in_other = depth_range, depth_range
            raise NotImplementedError()

        # UnProject
        if not load_latest:
            id_ref, nbr = 0, self.settings.nerf.rand_rays
        else:
            id_ref, nbr = 0, self.settings.nerf.rand_rays

        from source.utils.camera import pose_inverse_4x4
        from source.utils.geometry.batched_geometry_utils import batch_backproject_to_3d, batch_project
        from source.training.core.sampling_strategies import sample_rays
        from analysis.utils_evaldepth import compute_depth_errors
        bottom = torch.from_numpy(np.array([0, 0, 0, 1])).to(self.device).reshape(1, 1, -1).repeat(B, 1, 1)
        poses_w2c = data_dict.poses_w2c.detach()
        poses_w2c = torch.cat((poses_w2c, bottom), axis=1)
        poses_c2w = pose_inverse_4x4(poses_w2c)

        intr = data_dict.intr

        intr_ref = intr[id_ref]
        pose_w2c_ref = poses_w2c[id_ref]
        pose_c2w_ref = poses_c2w[id_ref]

        # project ref points to 3d ==> to create pseudo gt
        pixels_ref, _ = sample_rays(
            H, W, nbr=nbr,
            fraction_in_center=self.settings.sampled_fraction_in_center, seed=0
        )
        pixels_ref = pixels_ref.cuda()
        ret_ref = self.net.render_image_at_specific_pose_and_rays(self.settings, data_dict, depth_range_in_self,
                                                                  pose_w2c_ref[:3], intr=intr_ref, H=H, W=W,
                                                                  pixels=pixels_ref, mode='val', iter=self.iteration)
        depth_ref = ret_ref.depth.squeeze(0).squeeze(-1)
        depth_mono = data_dict['depth_est'][id_ref, pixels_ref[:, 1].long(), pixels_ref[:, 0].long()]
        depth_gt = data_dict['depth_gt'][id_ref, pixels_ref[:, 1].long(), pixels_ref[:, 0].long()]
        pts3d_in_w_from_ref = batch_backproject_to_3d(kpi=pixels_ref, di=depth_ref, Ki=intr_ref, T_itoj=pose_c2w_ref)

        '''
        import matplotlib
        matplotlib.use('TkAgg')
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            pts3d_in_w_from_ref[:, 0].cpu().numpy(),
            pts3d_in_w_from_ref[:, 1].cpu().numpy(),
            pts3d_in_w_from_ref[:, 2].cpu().numpy(),
            s=np.ones_like(pts3d_in_w_from_ref[:, 2].cpu().numpy()),
        )
        def set_axes_equal(ax):
            """
            Make axes of 3D plot have equal scale so that spheres appear as spheres,
            cubes as cubes, etc.

            Input
              ax: a matplotlib axis, e.g., as output from plt.gca().
            """

            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5 * max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        set_axes_equal(ax)
        plt.show()
        '''

        pts3d_in_w_from_ck_all = list()
        corres_gts_all, corres_est_all, corres_nerf_all = list(), list(), list()
        for k in range(1, self.settings.train_sub):
            pose_w2c_at_sampled = poses_w2c[k]
            pose_c2w_at_sampled = poses_c2w[k]
            intr_at_sampled = intr_ref.clone()

            pts_in_sampled_img, pseudo_gt_depth_in_sampled_img = batch_project(pts3d_in_w_from_ref,
                                                                               T_itoj=pose_w2c_at_sampled,
                                                                               Kj=intr_at_sampled,
                                                                               return_depth=True)
            ret_ck = self.net.render_image_at_specific_pose_and_rays(self.settings, data_dict, depth_range_in_self,
                                                                     pose_w2c_at_sampled[:3], intr=intr_at_sampled, H=H, W=W,
                                                                     pixels=pts_in_sampled_img, mode='val',
                                                                     iter=self.iteration)
            depth_ck = ret_ck.depth.squeeze(0).squeeze(-1)
            pts3d_in_w_from_ck = batch_backproject_to_3d(
                kpi=pts_in_sampled_img, di=depth_ck, Ki=intr_at_sampled, T_itoj=pose_c2w_at_sampled)
            pts3d_in_w_from_ck_all.append(pts3d_in_w_from_ck)

            corres_gts = data_dict['corres_gts_root2others'][k - 1][:, pixels_ref[:, 1].long(),
                        pixels_ref[:, 0].long()].T  # start from index 1
            corres_est = data_dict['corres_est_root2others'][k - 1][:, pixels_ref[:, 1].long(),
                        pixels_ref[:, 0].long()].T  # start from index 1

            corres_gts_all.append(corres_gts), corres_est_all.append(corres_est), corres_nerf_all.append(pts_in_sampled_img)

        corres_gts_all = torch.stack(corres_gts_all, dim=0)
        corres_est_all = torch.stack(corres_est_all, dim=0)
        corres_nerf_all = torch.stack(corres_nerf_all, dim=0)

        corres_sel = (torch.sum(corres_gts_all, dim=2) != torch.nan) * \
              (corres_nerf_all[:, :, 0] <= W - 1) * (corres_nerf_all[:, :, 0] > 0) * \
              (corres_nerf_all[:, :, 1] <= H - 1) * (corres_nerf_all[:, :, 1] > 0)

        pts3d_in_w_from_ck_all = torch.stack(pts3d_in_w_from_ck_all, dim=0)
        pts3d_diff = pts3d_in_w_from_ck_all - pts3d_in_w_from_ref.unsqueeze(0)
        pts3d_diff = torch.sqrt(torch.sum(pts3d_diff ** 2, dim=-1) + 1e-10) / depth_ref.unsqueeze(0)

        th_dists = np.linspace(np.log(0.05), np.log(0.0005), 200)
        th_dists = np.exp(th_dists)
        th_number, silog_a05_monos, silog_a05_nerfs, val_percent = 2, list(), list(), list()
        px1_corresest, px1_nerfest = list(), list()
        for th_dist in th_dists:
            val_observation = torch.sum(pts3d_diff < th_dist, dim=0) >= th_number

            silog_a05_mono = compute_depth_errors(depth_gt[val_observation], depth_mono[val_observation])
            silog_a05_nerf = compute_depth_errors(depth_gt[val_observation], depth_ref[val_observation])
            silog_a05_monos.append(np.array([silog_a05_mono[0], silog_a05_mono[7]]))
            silog_a05_nerfs.append(np.array([silog_a05_nerf[0], silog_a05_nerf[7]]))
            val_percent.append(torch.sum(val_observation).item() / len(val_observation))

            corres_gts_all_curf = corres_gts_all[corres_sel * val_observation.unsqueeze(0), :]
            corres_est_all_curf = corres_est_all[corres_sel * val_observation.unsqueeze(0), :]
            corres_nerf_all_curf = corres_nerf_all[corres_sel * val_observation.unsqueeze(0), :]

            px1_est = (torch.sum((corres_gts_all_curf - corres_est_all_curf) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_nerf = (torch.sum((corres_gts_all_curf - corres_nerf_all_curf) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_corresest.append(px1_est), px1_nerfest.append(px1_nerf)

        silog_a05_monos, silog_a05_nerfs = np.stack(silog_a05_monos, axis=0), np.stack(silog_a05_nerfs, axis=0)
        px1_corresest, px1_nerfest = np.array(px1_corresest), np.array(px1_nerfest)

        density_queries = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        to_plot = dict()
        for density_query in density_queries:
            closes_idx = np.argmin(np.abs(np.array(val_percent) - density_query))
            silog_monos, silog_nerfs = silog_a05_monos[closes_idx, 0], silog_a05_nerfs[closes_idx, 0]
            a05_monos, a05_nerfs = silog_a05_monos[closes_idx, 1], silog_a05_nerfs[closes_idx, 1]
            px1_from_corres, px1_from_nerf = px1_corresest[closes_idx], px1_nerfest[closes_idx]
            to_plot[density_query] = {
                'silog_monos': silog_monos, 'silog_nerfs': silog_nerfs,
                'a05_monos': a05_monos, 'a05_nerfs': a05_nerfs,
                'px1_from_corres': px1_from_corres, 'px1_from_nerf': px1_from_nerf
            }

        if np.isnan(silog_a05_monos).sum() + np.isnan(silog_a05_nerfs).sum() + np.isnan(px1_corresest).sum() + np.isnan(px1_nerfest).sum() > 50:
            return None, to_plot
        else:
            # import matplotlib
            # matplotlib.use('TkAgg')
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            axs[0, 0].plot(th_dists, silog_a05_monos[:, 0])
            axs[0, 0].plot(th_dists, silog_a05_nerfs[:, 0])
            axs[0, 0].set_xlabel("Triang Th in Meter")
            axs[0, 0].set_ylabel("Silog")
            axs[0, 0].legend(['Mono', 'MultiView'])
            axs[0, 0].set_title(self.settings.scene)

            axs[0, 1].plot(th_dists, silog_a05_monos[:, 1])
            axs[0, 1].plot(th_dists, silog_a05_nerfs[:, 1])
            axs[0, 1].set_xlabel("Triang Th in Meter")
            axs[0, 1].set_ylabel("A05")
            axs[0, 1].legend(['Mono', 'MultiView'])
            axs[0, 1].set_title(self.settings.scene)

            axs[1, 0].plot(th_dists, px1_corresest)
            axs[1, 0].plot(th_dists, px1_nerfest)
            axs[1, 0].set_xlabel("Triang Th in Meter")
            axs[1, 0].set_ylabel("Px1")
            axs[1, 0].legend(['CorresEst', 'MultiView'])
            axs[1, 0].set_title(self.settings.scene)

            axs[1, 1].plot(th_dists, np.array(val_percent) * 100)
            axs[1, 1].set_xlabel("Triang Th")
            axs[1, 1].set_ylabel("Density Percent")

            # image = tensor2rgb(self.train_data.all['image'], viewind=0)
            # th_dist_depth_vls = 0.03
            # val_observation = torch.sum(pts3d_diff < th_dist_depth_vls, dim=0) >= th_number
            # axs[2, 0].imshow(image)
            # axs[2, 0].scatter(pixels_ref[val_observation, 0].cpu().numpy(), pixels_ref[val_observation, 1].cpu().numpy(), 1)
            # axs[2, 1].imshow(image)

            # plt.show()
            to_plot_PIL_path = os.path.join(self.writer.log_dir, '{}_error_iter_{}.jpg'.format(self.writer.log_dir.split('/')[-1], str(self.iteration).zfill(10)))
            plt.savefig(to_plot_PIL_path, bbox_inches='tight')
            plt.close()

            return Image.open(to_plot_PIL_path), to_plot

    def multiview_bundle_adjustment_pose(self):
        from source.training.core.triangulation_loss import TriangulationLoss
        for x in self.loss_module.loss_modules:
            if isinstance(x, TriangulationLoss):
                break
        assert isinstance(x, TriangulationLoss)
        x.multiview_bundle_adjustment_pose(self, self.train_data.all)

    def run(self, load_latest: bool=True, make_validation_first: bool=False):
        """
        Main training loop function
        Here, load the whole training data for each iteration!
        """
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.settings.resume_snapshot is not None:
            # this is where it should load the weights of model only!!!
            checkpoint_path, weights = load_checkpoint(checkpoint=self.settings.resume_snapshot)
            self.load_state_dict(weights['state_dict'], do_partial_load=True)
            if 'nerf.progress' not in weights['state_dict']:
                self.net.nerf.progress.data.fill_(1.)

        self.just_started = True
        if load_latest:
            self.load_snapshot(ignore_fields=None)

        if make_validation_first and self.iteration == 0:
            self.inference()
            
        self.set_train_mode()
        self.summary_board.reset_all()
        self.timer.reset()

        #  Pose
        if self.settings.stage == 1:
            self.multiview_bundle_adjustment_pose()
            return
        elif self.settings.stage == 2:
            posenet_path = os.path.join(self.writer.log_dir, 'pose_optimized.pth')
            self.pose_net.load_state_dict(torch.load(posenet_path), strict=True)
        else:
            raise NotImplemented()

        self.before_train()
        self.optimizer.zero_grad()
        if self.optimizer_pose is not None: self.optimizer_pose.zero_grad()

        self.iteration = 0
        while self.iteration < self.max_iteration:

            # if self.iteration > 80000:
            # if self.iteration > 5000:
            #     self.logger.info("Max Iteration %d Reached, Terminated!" % 80000)
            #     break

            self.iteration += 1

            # here loads the full scene for training, i.e. all the images
            data_dict = self.train_data.all
            data_dict['iter'] = self.iteration

            result_dict = self.train_iteration(data_dict)  # where the loss is computed, gradients backpropagated and so on,

            # logging
            if self.iteration % self.log_steps == 0 or self.iteration == 1:
                summary_dict = self.summary_board.summary()
                message = '[Train] ' + get_log_string(
                    result_dict=summary_dict,
                    iteration=self.iteration,
                    max_iteration=self.max_iteration,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                del summary_dict['error_t / init'], summary_dict['error_R_rel_deg / init'], summary_dict['error_t_rel_deg / init']
                self.write_event('train', summary_dict, self.iteration)

            # snapshot, every certain number of iterations
            if self.iteration % self.snapshot_steps == 0:
                # snapshot saving and so on
                self.epoch = self.iteration
                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                self.delete_old_checkpoints()  # keep only the most recent set of checkpoints
            torch.cuda.empty_cache()

        
    @torch.no_grad()
    def make_result_dict(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                         loss: Dict[str, Any],metric: Dict[str, Any]=None, split: str='train'):
        stats_dict = {}
        for key,value in loss.items():
            if key=="all": continue
            stats_dict["loss_{}".format(key)] = value
        if metric is not None:
            for key,value in metric.items():
                stats_dict["{}".format(key)] = value
        return stats_dict


    @torch.no_grad()
    def val_step(self, iteration: int, data_dict: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: 
        data_dict = edict(data_dict)
        H, W = data_dict.image.shape[-2:] 
        plotting_dict = {}

        output_dict = self.net.forward(self.settings,data_dict,mode="val",iter=self.iteration)  
        # will render the full image
        output_dict['mse'], output_dict['mse_fine'] = compute_mse_on_rays(data_dict, output_dict)

        # to compute the loss:
        # poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='val')  
        # data_dict.poses_w2c = poses_w2c
        # loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss\
        #     (self.settings, data_dict, output_dict, iteration=iteration, mode="val")

        results_dict = self.make_result_dict(self.settings, data_dict, output_dict, loss={}, split='val')
        # results_dict.update(stats_dict)
        # results_dict['loss'] = loss_dict['all']
        
        results_dict['best_value'] = - results_dict['PSNR_fine'] if 'PSNR_fine' in results_dict.keys() \
            else - results_dict['PSNR']
        
        # run some evaluations
        gt_image = data_dict.image.reshape(-1, 3, H, W)
        
        # coarse prediction
        pred_rgb_map = output_dict.rgb.reshape(-1, H, W, 3).permute(0,3,1,2)  # (B, 3, H, W)
        ssim = ssim_loss(pred_rgb_map, gt_image).item()
        lpips = self.lpips_loss(pred_rgb_map*2-1, gt_image*2-1).item()

        results_dict['ssim'] = ssim
        results_dict['lpips'] = lpips

        if 'fg_mask' in data_dict.keys():
            results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image, 
                                                       self.lpips_loss, suffix=''))

        if 'rgb_fine' in output_dict.keys():
            pred_rgb_map = output_dict.rgb_fine.reshape(-1, H, W, 3).permute(0,3,1,2)
            ssim = ssim_loss(pred_rgb_map, gt_image).item()
            lpips = self.lpips_loss(pred_rgb_map*2-1, gt_image*2-1).item()

            results_dict['ssim_fine'] = ssim
            results_dict['lpips_fine'] = lpips
            
            if 'fg_mask' in data_dict.keys():
                results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image, 
                                                           self.lpips_loss, suffix='_fine'))

        if iteration < 5 or (iteration % 4 == 0): 
            plotting_dict_ = self.visualize(self.settings, data_dict, output_dict, step=iteration, split="val")
            plotting_dict.update(plotting_dict_)
        return output_dict, results_dict, plotting_dict 

    def eval_after_training(self, load_best_model: bool=False, plot: bool=False, 
                            save_ind_files: bool=False):
        """ Run final evaluation on the test set. Computes novel-view synthesis performance. 
        When the poses were optimized, also computes the pose registration error. Optionally, one can run
        test-time pose optimization to factor out the pose error from the novel-view synthesis performance. 
        """
        self.logger.info('DOING EVALUATION')
        model_name = self.settings.model
        args = self.settings
        args.expname = self.settings.script_name

        # the loss is redefined here as only the photometric one, in case the 
        # test-time photometric optimization is used
        args.loss_type = 'photometric'
        args.loss_weight.render = 0.

        if load_best_model:
            checkpoint_path = '{}/{}/model_best.pth.tar'.format(self._base_save_dir, self.settings.project_path)
            self.logger.info('Loading {}'.format(checkpoint_path))
            weights =  torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if hasattr(self, 'load_state_dict'):
                self.load_state_dict(weights['state_dict'])
            else:
                self.net.load_state_dict(weights['state_dict'], strict=True)
            if 'iteration' in weights.keys():
                self.iteration = weights['iteration']

        # define name of experiment
        dataset_name = args.dataset
        if hasattr(args, 'factor') and args.factor != 1:
            dataset_name += "_factor_" + str(args.factor)
        elif hasattr(args, 'llff_img_factor') and args.llff_img_factor != 1:
            dataset_name += "_factor_" + str(args.llff_img_factor)
        if hasattr(args, 'resize') and args.resize:
            dataset_name += "_{}x{}".format(args.resize[0], args.resize[1]) if len(args.resize) == 2 else "_" + str(args.resize)
        elif hasattr(args, 'resize_factor') and args.resize_factor:
            dataset_name += "_resizefactor_" + str(args.resize_factor)
        
        # define the output directory where the metrics (and qualitative results) will be stored
        if self.debug:
            out_dir = os.path.join(args.env.eval_dir + '_debug', dataset_name)
        else:
            out_dir = os.path.join(args.env.eval_dir, dataset_name)

        if args.train_sub is None:
            out_dir = os.path.join(out_dir, 'all_training_views')
        else:
            out_dir = os.path.join(out_dir, f'{args.train_sub}_training_views')
            
        out_dir = os.path.join(out_dir, args.scene)
        out_dir = os.path.join(out_dir, self.settings.module_name_for_eval)
        extra_out_dir = os.path.join(out_dir, args.expname + '_{}'.format(self.iteration))  # to save qualitative figures

        self.logger.critical('Experiment: {} / {}'.format(self.settings.module_name, args.expname))
        self.logger.critical("saving results to {}...".format(out_dir))
        os.makedirs(out_dir, exist_ok=True)

        # load the test step
        args.val_sub = None  
        self.load_dataset(args, eval_split='test')
        
        save_all = {}
        test_optim_options = [True, False] if model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] else [False]
        for test_optim in test_optim_options:
            self.logger.info('test pose optim : {}'.format(test_optim))
            args.optim.test_photo = test_optim

            possible_to_plot = True
            if test_optim is False and model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                possible_to_plot = False
            results_dict = self.evaluate_full(args, plot=plot and possible_to_plot, 
                                              save_ind_files=save_ind_files and possible_to_plot, 
                                              out_scene_dir=extra_out_dir)

            if test_optim:
                save_all['w_test_optim'] = results_dict
            elif model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                save_all['without_test_optim'] = results_dict
            else:
                # nerf 
                save_all = results_dict
        
        save_all['iteration'] = self.iteration

        if load_best_model:
            name_file = '{}_best_model.txt'.format(args.expname)
        else:
            name_file = '{}.txt'.format(args.expname)
        self.logger.critical('Saving json file to {}/{}'.format(out_dir, name_file))
        with open("{}/{}".format(out_dir, name_file), "w+") as f:
            json.dump(save_all, f, indent=4)
        return 


    @torch.no_grad()
    def visualize(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                  step: int=0, split: str="train", eps: float=1e-10
                  ) -> Dict[str, Any]:
        """Creates visualization of renderings and gt. Here N is HW

        Attention:
            ground-truth image has shape (B, 3, H, W)
            rgb rendering has shape (B, H*W, 3)
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
            step (int, optional): Defaults to 0.
            split (str, optional): Defaults to "train".
        """
        plotting_stats = {}

        analysis_fig, analysis_statistics = self.analysis(load_latest=False)
        if analysis_fig is not None:
            analysis_fig = torch.from_numpy(np.array(analysis_fig)).float().permute([2, 0, 1]).unsqueeze(0) / 255.0
            plotting_stats[f'{split}_{step}/stats'] = analysis_fig

        _, _, H, W = data_dict['image'].shape
        nview, dummy_iteration = self.settings.train_sub, 10000
        fig_all = list()
        for idx_view in range(nview):
            rendered = self.net.render_image_at_specific_rays(opt, data_dict, img_idx=idx_view, iter=dummy_iteration, mode="eval")
            depth_nerf, depth_mono = rendered['depth'].view([1, 1, H, W]), data_dict['depth_est'][idx_view].view([1, 1, H, W])
            fig_depth_nerf, fig_depth_mono = tensor2disp(1/depth_nerf, vmax=1.0, viewind=0), tensor2disp(1/depth_mono, vmax=1.0, viewind=0)
            fig_rgb = tensor2rgb(data_dict['image'], viewind=idx_view)
            fig_all.append(np.concatenate(
                [np.array(fig_rgb), np.array(fig_depth_nerf), np.array(fig_depth_mono)], axis=0
            ))
        fig_all = np.concatenate(fig_all, axis=1)
        fig_all = torch.from_numpy(fig_all).float().permute([2, 0, 1]).unsqueeze(0) / 255.0
        plotting_stats[f'{split}_{step}/fig_all'] = fig_all

        plotting_stats['analysis_statistics'] = analysis_statistics
        return plotting_stats

    @torch.no_grad()
    def save_ind_files(self, save_dir: str, name: str, image: torch.Tensor, 
                       rendered_img: torch.Tensor, rendered_depth: torch.Tensor, 
                       depth_range: List[float]=None, depth_gt: torch.Tensor=None):
        """Save rendering and ground-truth data as individual files. 
        
        Args:
            save_dir (str): dir to save the images
            name (str): name of image (without extension .png)
            image (torch.Tensor): gt image of shape [1, 3, H, W]
            rendered_img (torch.Tensor): rendered image of shape [1, 3, H, W]
            rendered_depth (torch.Tensor): rendered depth of shape [1, H, W, 1]
            depth_range (list of floats): depth range for depth visualization
            depth_gt (torch.Tensor): gt depth of shape [1, H, W, 1]
        """
        rend_img_dir = os.path.join(save_dir, 'rendered_imgs')
        rend_depth_dir = os.path.join(save_dir, 'rendered_depths')
        gt_img_dir = os.path.join(save_dir, 'gt_imgs')
        gt_depth_dir = os.path.join(save_dir, 'gt_depths')
        if not os.path.exists(rend_img_dir):
            os.makedirs(rend_img_dir, exist_ok=True)
        if not os.path.exists(gt_img_dir):
            os.makedirs(gt_img_dir, exist_ok=True)
        if not os.path.exists(rend_depth_dir):
            os.makedirs(rend_depth_dir, exist_ok=True)
        if not os.path.exists(gt_depth_dir):
            os.makedirs(gt_depth_dir, exist_ok=True)

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8) # (B, H, W, 3), B is 1
        imageio.imwrite(os.path.join(gt_img_dir, name + '.png'), image)
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!

        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy() # [B,3, H,W] and then (H, W, 3)
        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_img_dir, name + '.png'), fine_pred_rgb_np_uint8)


        depth = rendered_depth[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W)
        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_depth_dir, name + '.png'), fine_pred_depth_colored)

        if depth_gt is not None:
            depth = depth_gt[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
            fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
            fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
            imageio.imwrite(os.path.join(gt_depth_dir, name + '.png'), fine_pred_depth_colored)
        return 

    @torch.no_grad()
    def visualize_eval(self, to_plot: List[Any], image: torch.Tensor, rendered_img: torch.Tensor, 
                       rendered_depth: torch.Tensor, rendered_depth_var: torch.Tensor, depth_range: List[float]=None):
        """Visualization for the test set"""

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8) # (B, H, W, 3), B is 1
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!
        depth = rendered_depth[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
        depth_var = rendered_depth_var[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy() # [B,3, H,W] and then (H, W, 3)

        
        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)

        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

        fine_pred_depth_var_colored = colorize_np(depth_var, append_cbar=False)
        fine_pred_depth_var_colored = (255 * fine_pred_depth_var_colored).astype(np.uint8)

        to_plot += [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, fine_pred_depth_var_colored]
        return 


    @torch.no_grad()
    def yield_sparse_estimate(self, load_latest: bool=True):
        if load_latest:
            self.load_snapshot()

        # Switch to Evaluation Mode
        self.net.eval()

        for x in self.loss_module.loss_modules:
            if isinstance(x, TriangulationLoss):
                break
        assert isinstance(x, TriangulationLoss)

        x.inference_sparse_depth(renderer=self, data_dict=self.train_data.all)
        a = 1
        data_dict['iter'] = self.iteration
        B, _, H, W = data_dict['image'].shape


        _, _, H, W = data_dict['image'].shape
        idx_root = 0

        # Render Pose
        poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='train')  # is it world to camera
        data_dict.poses_w2c = poses_w2c
        poses_w2c = padding_pose(poses_w2c)
        intirnsic44 = padding_pose(data_dict.intr)

        # Sampling
        pixels_root, _ = sample_rays(
            H, W, nbr=self.settings.nerf.rand_rays,
            fraction_in_center=self.settings.sampled_fraction_in_center,
        )
        pixels_root, depth_range = pixels_root.cuda(), acquire_depth_range(self.settings, data_dict)


        # UnProject
        if not load_latest:
            id_ref, nbr = 0, self.settings.nerf.rand_rays
        else:
            id_ref, nbr = 0, self.settings.nerf.rand_rays

        from source.utils.camera import pose_inverse_4x4
        from source.utils.geometry.batched_geometry_utils import batch_backproject_to_3d, batch_project
        from source.training.core.sampling_strategies import sample_rays
        from analysis.utils_evaldepth import compute_depth_errors
        bottom = torch.from_numpy(np.array([0, 0, 0, 1])).to(self.device).reshape(1, 1, -1).repeat(B, 1, 1)
        poses_w2c = data_dict.poses_w2c.detach()
        poses_w2c = torch.cat((poses_w2c, bottom), axis=1)
        poses_c2w = pose_inverse_4x4(poses_w2c)

        intr = data_dict.intr

        intr_ref = intr[id_ref]
        pose_w2c_ref = poses_w2c[id_ref]
        pose_c2w_ref = poses_c2w[id_ref]

        # project ref points to 3d ==> to create pseudo gt
        pixels_ref, _ = sample_rays(
            H, W, nbr=nbr,
            fraction_in_center=self.settings.sampled_fraction_in_center
        )
        pixels_ref = pixels_ref.cuda()
        ret_ref = self.net.render_image_at_specific_pose_and_rays(self.settings, data_dict, depth_range_in_self,
                                                                  pose_w2c_ref[:3], intr=intr_ref, H=H, W=W,
                                                                  pixels=pixels_ref, mode='val', iter=self.iteration)
        depth_ref = ret_ref.depth.squeeze(0).squeeze(-1)
        depth_mono = data_dict['depth_est'][id_ref, pixels_ref[:, 1].long(), pixels_ref[:, 0].long()]
        depth_gt = data_dict['depth_gt'][id_ref, pixels_ref[:, 1].long(), pixels_ref[:, 0].long()]
        pts3d_in_w_from_ref = batch_backproject_to_3d(kpi=pixels_ref, di=depth_ref, Ki=intr_ref, T_itoj=pose_c2w_ref)

        pts3d_in_w_from_ck_all = list()
        corres_gts_all, corres_est_all, corres_nerf_all = list(), list(), list()
        for k in range(1, self.settings.train_sub):
            pose_w2c_at_sampled = poses_w2c[k]
            pose_c2w_at_sampled = poses_c2w[k]
            intr_at_sampled = intr_ref.clone()

            pts_in_sampled_img, pseudo_gt_depth_in_sampled_img = batch_project(pts3d_in_w_from_ref,
                                                                               T_itoj=pose_w2c_at_sampled,
                                                                               Kj=intr_at_sampled,
                                                                               return_depth=True)
            ret_ck = self.net.render_image_at_specific_pose_and_rays(self.settings, data_dict, depth_range_in_self,
                                                                     pose_w2c_at_sampled[:3], intr=intr_at_sampled, H=H, W=W,
                                                                     pixels=pts_in_sampled_img, mode='val',
                                                                     iter=self.iteration)
            depth_ck = ret_ck.depth.squeeze(0).squeeze(-1)
            pts3d_in_w_from_ck = batch_backproject_to_3d(
                kpi=pts_in_sampled_img, di=depth_ck, Ki=intr_at_sampled, T_itoj=pose_c2w_at_sampled)
            pts3d_in_w_from_ck_all.append(pts3d_in_w_from_ck)

            sample_xx, sample_yy = torch.split(pts_in_sampled_img.unsqueeze(0), 1, dim=2)
            sample_xx, sample_yy = (sample_xx / (W - 1) - 0.5) * 2, (sample_yy / (H - 1) - 0.5) * 2
            pts_in_sampled_img_normed = torch.cat([sample_xx, sample_yy], dim=-1)

            corres_gts = data_dict['corres_gts_root2others'][k - 1][:, pixels_ref[:, 1].long(),
                        pixels_ref[:, 0].long()].T  # start from index 1
            corres_est = data_dict['corres_est_root2others'][k - 1][:, pixels_ref[:, 1].long(),
                        pixels_ref[:, 0].long()].T  # start from index 1

            corres_gts_all.append(corres_gts), corres_est_all.append(corres_est), corres_nerf_all.append(pts_in_sampled_img)

        corres_gts_all = torch.stack(corres_gts_all, dim=0)
        corres_est_all = torch.stack(corres_est_all, dim=0)
        corres_nerf_all = torch.stack(corres_nerf_all, dim=0)

        corres_sel = (torch.sum(corres_gts_all, dim=2) != torch.nan) * \
              (corres_nerf_all[:, :, 0] <= W - 1) * (corres_nerf_all[:, :, 0] > 0) * \
              (corres_nerf_all[:, :, 1] <= H - 1) * (corres_nerf_all[:, :, 1] > 0)

        pts3d_in_w_from_ck_all = torch.stack(pts3d_in_w_from_ck_all, dim=0)
        pts3d_diff = pts3d_in_w_from_ck_all - pts3d_in_w_from_ref.unsqueeze(0)
        pts3d_diff = torch.sqrt(torch.sum(pts3d_diff ** 2, dim=-1) + 1e-10) / depth_ref.unsqueeze(0)

        # from source.utils.geometry.batched_geometry_utils import to_homogeneous, from_homogeneous
        # pts3d_in_w_from_ck_all_prj = pts3d_in_w_from_ck_all @ intr_ref.transpose(-1, -2)
        # pts3d_in_w_from_ck_all_prj = from_homogeneous(pts3d_in_w_from_ck_all_prj)

        sample_xx, sample_yy = torch.split(pts3d_in_w_from_ck_all, 1, dim=2)
        sample_xx, sample_yy = (sample_xx / (W - 1) - 0.5) * 2, (sample_yy / (H - 1) - 0.5) * 2
        pts3d_in_w_from_ck_all_normed = torch.cat([sample_xx, sample_yy], dim=-1)

        th_dists = np.linspace(np.log(0.05), np.log(0.0005), 200)
        th_dists = np.exp(th_dists)
        th_number, silog_a05_monos, silog_a05_nerfs, val_percent = 2, list(), list(), list()
        px1_corresest, px1_nerfest = list(), list()
        for th_dist in th_dists:
            val_observation = torch.sum(pts3d_diff < th_dist, dim=0) >= th_number

            silog_a05_mono = compute_depth_errors(depth_gt[val_observation], depth_mono[val_observation])
            silog_a05_nerf = compute_depth_errors(depth_gt[val_observation], depth_ref[val_observation])
            silog_a05_monos.append(np.array([silog_a05_mono[0], silog_a05_mono[7]]))
            silog_a05_nerfs.append(np.array([silog_a05_nerf[0], silog_a05_nerf[7]]))
            val_percent.append(torch.sum(val_observation).item() / len(val_observation))

            corres_gts_all_curf = corres_gts_all[corres_sel * val_observation.unsqueeze(0), :]
            corres_est_all_curf = corres_est_all[corres_sel * val_observation.unsqueeze(0), :]
            corres_nerf_all_curf = corres_nerf_all[corres_sel * val_observation.unsqueeze(0), :]

            px1_est = (torch.sum((corres_gts_all_curf - corres_est_all_curf) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_nerf = (torch.sum((corres_gts_all_curf - corres_nerf_all_curf) ** 2, dim=1).sqrt() < 1).float().mean().item()
            px1_corresest.append(px1_est), px1_nerfest.append(px1_nerf)

        silog_a05_monos, silog_a05_nerfs = np.stack(silog_a05_monos, axis=0), np.stack(silog_a05_nerfs, axis=0)
        px1_corresest, px1_nerfest = np.array(px1_corresest), np.array(px1_nerfest)

        if np.isnan(silog_a05_monos).sum() + np.isnan(silog_a05_nerfs).sum() + np.isnan(px1_corresest).sum() + np.isnan(px1_nerfest).sum() > 50:
            return None
        else:
            # import matplotlib
            # matplotlib.use('TkAgg')
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            axs[0, 0].plot(th_dists, silog_a05_monos[:, 0])
            axs[0, 0].plot(th_dists, silog_a05_nerfs[:, 0])
            axs[0, 0].set_xlabel("Triang Th in Meter")
            axs[0, 0].set_ylabel("Silog")
            axs[0, 0].legend(['Mono', 'MultiView'])
            axs[0, 0].set_title(self.settings.scene)

            axs[0, 1].plot(th_dists, silog_a05_monos[:, 1])
            axs[0, 1].plot(th_dists, silog_a05_nerfs[:, 1])
            axs[0, 1].set_xlabel("Triang Th in Meter")
            axs[0, 1].set_ylabel("A05")
            axs[0, 1].legend(['Mono', 'MultiView'])
            axs[0, 1].set_title(self.settings.scene)

            axs[1, 0].plot(th_dists, px1_corresest)
            axs[1, 0].plot(th_dists, px1_nerfest)
            axs[1, 0].set_xlabel("Triang Th in Meter")
            axs[1, 0].set_ylabel("Px1")
            axs[1, 0].legend(['CorresEst', 'MultiView'])
            axs[1, 0].set_title(self.settings.scene)

            axs[1, 1].plot(th_dists, np.array(val_percent) * 100)
            axs[1, 1].set_xlabel("Triang Th")
            axs[1, 1].set_ylabel("Density Percent")

            # image = tensor2rgb(self.train_data.all['image'], viewind=0)
            # th_dist_depth_vls = 0.03
            # val_observation = torch.sum(pts3d_diff < th_dist_depth_vls, dim=0) >= th_number
            # axs[2, 0].imshow(image)
            # axs[2, 0].scatter(pixels_ref[val_observation, 0].cpu().numpy(), pixels_ref[val_observation, 1].cpu().numpy(), 1)
            # axs[2, 1].imshow(image)

            # plt.show()
            to_plot_PIL_path = os.path.join(self.writer.log_dir, '{}_error_iter_{}.jpg'.format(self.writer.log_dir.split('/')[-1], str(self.iteration).zfill(10)))
            plt.savefig(to_plot_PIL_path, bbox_inches='tight')
            plt.close()

            return Image.open(to_plot_PIL_path)