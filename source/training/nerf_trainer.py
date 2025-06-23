
import os, sys, time, inspect, urllib
import imageio
import torch
from easydict import EasyDict as edict

from source.training import base
import source.models.renderer as renderer
from source.models.flow_net import FlowSelectionWrapper
from source.training.core.sampling_strategies import RaySamplingStrategy
from typing import Any, Dict, Tuple, List

class NerfTrainerPerScene(base.PerSceneTrainer):
    """ Base class for training and evaluating a NeRF model, considering fixed ground-truth poses. """

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)
        self.init_for_training(opt)

    def init_for_training(self, opt: Dict[str, Any]):
        # define and load the datasets
        # this needs to be first for the joint pose-NeRF refinements, since there will be pose parameters 
        # for each of the training images. 
        self.load_dataset(opt)

        # define the model
        self.build_networks(opt)

        # define optimizer and scheduler
        self.setup_optimizer(opt)

        # define losses
        self.define_loss_module(opt)

        del self.flow_net
        del self.monodepth_net
        return

    def plot_training_set(self):
        value = self.train_data.all.image   # (B, 3, H, W)
        self.writer.add_images(f'train/training_images', value, 1, dataformats='NCHW')
        return

    def load_dataset(self,opt: Dict[str, Any],eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        
        # prefetch all training data
        self.train_data.prefetch_all_data()
        self.train_data.all = edict(self.to_cuda(self.train_data.all))  
        # contains the data corresponding to the entire scene
        # important keys are 'image', 'pose' (w2c), 'intr'
        self.plot_training_set()
        
        
        # pixels/ray sampler
        self.sampling_strategy = RaySamplingStrategy(opt, data_dict=self.train_data.all, device=self.device)


        self.logger.info('Depth type {}'.format(opt.nerf.depth.param))
        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = self.train_data.all.depth_range[0]
        self.logger.info('depth range {} to {}'.format(depth_range[0], depth_range[1]))
        return

    def build_nerf_net(self, opt: Dict[str, Any]):
        self.net = renderer.Graph(opt, self.device)
        return
    
    def build_correspondence_net(self, opt: Dict[str, Any]):
        self.flow_net = FlowSelectionWrapper(ckpt_path=opt.flow_ckpt_path,
                                             backbone=opt.flow_backbone,
                                             batch_size=opt.flow_batch_size,
                                             num_views=len(self.train_data)).to(self.device)
        return

    def build_monodepth_net(self,  opt: Dict[str, Any]):
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        if opt.mondepth_backbone == 'ZoeDepth':
            ZoeDepth_root = os.path.join(proj_root, 'third_party', 'ZoeDepth')
            sys.path.insert(0, ZoeDepth_root)
            from zoedepth.models.builder import build_model
            from zoedepth.utils.config import get_config
            conf = get_config("zoedepth_nk", "infer")
            conf['pretrained_resource'] = "local::" + os.path.join(proj_root, 'checkpoint', 'ZoeD_M12_NK.pt')
            self.monodepth_net = build_model(conf).eval().to(self.device)
        elif opt.mondepth_backbone == 'ZeroDepth':
            vidar_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
            vidar_root = os.path.join(vidar_root, 'third_party', 'vidar')
            sys.path.insert(0, vidar_root)

            from vidar.utils.config import read_config
            from vidar.utils.setup import setup_network

            cfg_url = "https://raw.githubusercontent.com/TRI-ML/vidar/main/configs/papers/zerodepth/hub_zerodepth.yaml"
            cfg = urllib.request.urlretrieve(cfg_url, "zerodepth_config.yaml")
            cfg = read_config("zerodepth_config.yaml")
            model = setup_network(cfg.networks.perceiver)
            model.eval()
            model = model.to(self.device)

            url = "https://tri-ml-public.s3.amazonaws.com/github/vidar/models/ZeroDepth_unified.ckpt"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            state_dict = {k.replace("module.networks.define.", ""): v for k, v in state_dict["state_dict"].items()}
            model.load_state_dict(state_dict, strict=True)
            self.monodepth_net = model

        elif opt.mondepth_backbone == 'Metric3DDepth':
            metric3D_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
            metric3D_root = os.path.join(metric3D_root, 'third_party', 'Metric3D')
            sys.path.insert(0, metric3D_root)
            try:
                from mmcv.utils import Config, DictAction
            except:
                from mmengine import Config, DictAction
            from mono.model.monodepth_model import get_configured_monodepth_model
            from mono.utils.running import load_ckpt
            from mono.utils.do_test import transform_test_data_scalecano, get_prediction
            config_path = os.path.join(metric3D_root, 'mono', 'configs', 'HourglassDecoder', 'convlarge.0.3_150.py')
            ckpt_path = os.path.join(metric3D_root, '..', '..', 'checkpoint', 'convlarge_hourglass_0.3_150_step750k_v1.1.pth')
            cfg = Config.fromfile(config_path)
            model = get_configured_monodepth_model(cfg, )
            model = torch.nn.DataParallel(model)
            model, _, _, _ = load_ckpt(ckpt_path, model, strict_match=False)
            model.eval()
            model = model.to(self.device)
            class Metric3D:
                def __init__(self, model, cfg):
                    self.model = model
                    # self.mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None].to(device).float()
                    # self.std = torch.tensor([58.395, 57.12, 57.375])[:, None, None].to(device).float()
                    self.cfg = cfg
                def infer(self, rgb, intrinsic):
                    """
                    Args:
                        rgb: cv2 bgr image. [H, W, 3]
                        intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
                    """
                    normalize_scale = self.cfg.data_basic.depth_range[1]
                    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb, intrinsic, self.cfg.data_basic)
                    pred_depth, pred_depth_scale, scale = get_prediction(
                        model=model,
                        input=rgb_input,
                        cam_model=cam_models_stacks,
                        pad_info=pad,
                        scale_info=label_scale_factor,
                        gt_depth=None,
                        normalize_scale=normalize_scale,
                        ori_shape=[rgb.shape[0], rgb.shape[1]],
                    )
                    pred_depth = pred_depth.unsqueeze(0)
                    return pred_depth
            self.monodepth_net = Metric3D(model, cfg)

        else:
            raise ValueError()

    def get_colmap_triangulation(self):
        """Triangulate 3D points using the ground-truth poses and correspondences extracted
        by multiple alternative matches. This is basically what is done in DS-NeRF to obtain the
        colmap depth maps. This is used for comparison but does not correspond to a realistic scenario, 
        because ground-truth poses are used. 
        """
        matcher = self.settings.matcher_for_colmap_triang
        directory = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        if matcher == 'sp_sg':
            colmap_depth_map, colmap_conf_map = compute_triangulation_sp_sg\
                (opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
        elif matcher == 'pdcnet':
            colmap_depth_map, colmap_conf_map = compute_triangulation_pdcnet\
                (opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
        else:
            raise ValueError
        
        # adding to the data_dict
        self.train_data.all.colmap_depth = colmap_depth_map.to(self.device) 
        self.train_data.all.colmap_conf = colmap_conf_map.to(self.device)         

        if 'depth_gt' in self.train_data.all.keys():
            # look at the average error in depth of colmap
            depth_gt = self.train_data.all.depth_gt
            valid_depth_gt = self.train_data.all.valid_depth_gt
            colmap_depth = self.train_data.all.colmap_depth
            valid_colmap_depth = colmap_depth.gt(1e-6)

            mask = valid_depth_gt & valid_colmap_depth
            depth_gt = depth_gt[mask]
            colmap_depth = colmap_depth[mask]
            error = torch.abs(depth_gt - colmap_depth).mean()
            self.logger.info('colmap depth error {}'.format(error))
            self.write_event('train', {'colmap_depth_err': error}, self.iteration)

        '''
        if self.settings.load_colmap_depth and self.settings.debug:
            for i in range(self.train_data.all.image.shape[0]):
                for j in range(i+1, self.train_data.all.image.shape[0]):
                    verify_colmap_depth(data_dict=self.train_data.all, poses_w2c=self.train_data.all.pose, 
                                        index_0=i, index_1=j)
        '''
        return 

    def build_networks(self, opt: Dict[str, Any]):
        self.logger.info("building networks...")
        self.build_nerf_net(opt)

        if opt.load_colmap_depth:
            self.get_colmap_triangulation()

        if opt.use_flow:
            self.build_correspondence_net(opt)
        return 

    def setup_optimizer(self,opt: Dict[str, Any]):
        self.logger.info("setting up optimizer for NeRF...")

        optim = torch.optim.Adam(params=self.net.nerf.parameters(), 
                                 lr=opt.optim.lr, betas=(0.9, 0.999))
        if opt.nerf.fine_sampling:
            optim.add_param_group(dict(params=self.net.nerf_fine.parameters(),
                                       lr=opt.optim.lr, betas=(0.9, 0.999))) 

        self.register_optimizer(optim)

        # set up scheduler
        if opt.optim.sched:
            self.logger.info('setting up scheduler for NeRF...')
            if opt.optim.start_decrease > 0:
                sched = exponentiel_lr_starting_at_x(opt, self.optimizer)
            else:
                # normal schedular, starting from the first iteration
                scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
                if opt.optim.lr_end:
                    assert(opt.optim.sched.type=="ExponentialLR")
                    max_iter = opt.optim.max_iter if hasattr(opt.optim, 'max_iter') else opt.max_iter
                    if opt.optim.lr != 0.0:
                        opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./max_iter)
                    else:
                        opt.optim.sched.gamma = 0.0
                kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
                sched = scheduler(optimizer=self.optimizer, **kwargs)
            self.register_scheduler(sched)
        return 
        
    def train_step(self, iteration: int, data_dict: Dict[str, Any]
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Forward pass of the training step. Loss computation
        Args:
            iteration (int): 
            data_dict (edict): Scene data. dict_keys(['idx', 'image', 'intr', 'pose'])
                                image: tensor of images with shape (B, 3, H, W)
                                intr: tensor of intrinsic matrices of shape (B, 3, 3)
                                pose: tensor of ground-truth w2c poses of shape (B, 3, 4)
                                idx: indices of images, shape is (B)

        Returns:
            output_dict: output from the renderer
            results_dict: dict to log, also contains the loss for back-propagation
            plotting_dict: dict with figures to log to tensorboard. 
        """
        plot = (iteration % self.settings.vis_steps == 0) or (iteration == 1)
        do_log = iteration % self.settings.log_steps == 0 or iteration == 1
        output_dict = dict()

        assert self.settings.loss_type == 'triangulation'
        loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss \
            (self.settings, data_dict, output_dict=dict(), mode="train", plot=plot, iteration=iteration, renderer=self, do_log=do_log)

        results_dict = stats_dict
        results_dict['loss'] = loss_dict['all']  # the actual loss, used to back-propagate

        if do_log:
            # relatively heavy computation, so do not do it at each step
            # for example, pose alignement + registration evaluation
            results_dict.update(self.make_results_dict_low_freq(self.settings, data_dict, stats_dict, loss_dict))
            error_t, error_R_rel_deg, error_t_rel_deg = results_dict['error_t'], results_dict['error_R_rel_deg'], results_dict['error_t_rel_deg']
            del results_dict['error_t'], results_dict['error_R_rel_deg'], results_dict['error_t_rel_deg']
            results_dict.update({
                'error_t / init': "%.3f / %.3f" % (error_t, self.initial_pose_error['error_t']),
                'error_R_rel_deg / init': "%.3f / %.3f" % (error_R_rel_deg, self.initial_pose_error['error_R_rel_deg']),
                'error_t_rel_deg / init': "%.3f / %.3f" % (error_t_rel_deg, self.initial_pose_error['error_t_rel_deg']),
                'consumed_minutes': (time.time() - self.training_start_time) / 60
            })

        if plot:
            # render the full image
            with torch.no_grad():
                self.net.eval()
                # will render the full image
                plotting_dict = self.visualize(
                    self.settings, data_dict, None, split='train'
                )
                self.net.train()

                analysis_statistics = plotting_dict['analysis_statistics']
                keys = ['silog_improve', 'a05_improve', 'px1_improve']

                for key in keys:
                    for density in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
                        if key == 'silog_improve':
                            result_dict_key = "silog_improve/{}".format(density)
                            results_dict[result_dict_key] = analysis_statistics[density]['silog_nerfs'] - analysis_statistics[density]['silog_monos']
                        elif key == 'a05_improve':
                            result_dict_key = "a05_improve/{}".format(density)
                            results_dict[result_dict_key] = analysis_statistics[density]['a05_nerfs'] - analysis_statistics[density]['a05_monos']
                        elif key == 'px1_improve':
                            result_dict_key = "px1_improve/{}".format(density)
                            results_dict[result_dict_key] = analysis_statistics[density]['px1_from_nerf'] - analysis_statistics[density]['px1_from_corres']
                del plotting_dict['analysis_statistics']

        return output_dict, results_dict, plotting_dict 

    @torch.no_grad()
    def make_result_dict(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                         loss: Dict[str, Any], metric: Dict[str, Any]=None, 
                         split: str='train') -> Dict[str, Any]:
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = super().make_result_dict(opt,data_dict, output_dict,loss,metric=metric,split=split)
        return stats_dict

    @torch.no_grad()
    def make_results_dict_low_freq(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],loss: Dict[str, Any],
                                   metric: Dict[str, Any]=None, idx_optimized_pose: List[int]=None, split:str="train"):
        """
        Here corresponds to dictionary which will be saved in tensorboard and also logged"""
        return {}

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        """Get ground-truth (canonical) camera poses"""
        pose_GT = self.train_data.get_all_camera_poses(opt).to(self.device)
        return None, pose_GT

    @torch.no_grad()
    def evaluate_full(self, opt: Dict[str, Any], plot: bool=False, 
                      save_ind_files: bool=False, out_scene_dir: str=''):
        """
        Does the actual evaluation here on the test set. Important that opt is given as variable to change
        the test time optimization input. 
        Args:
            opt (edict): settings
            plot (bool, optional): Defaults to False.
            save_ind_files (bool, optional): Defaults to False
            out_scene_dir (str, optional): Path to dir, to save visualization if plot is True. Defaults to ''.
        Returns: dict with evaluation results
        """
        self.net.eval()
        # loader = tqdm.tqdm(self.val_loader,desc="evaluating",leave=False)
        res = []
        if plot:
            os.makedirs(out_scene_dir, exist_ok=True)

        results_dict = {'single': {}}
        for i, batch in enumerate(self.val_loader):
            # batch contains a single image here
            data_dict = edict(self.to_cuda(batch))
            
            file_id = os.path.basename(batch['rgb_path'][0]).split('.')[0] if 'rgb_path' \
                in batch.keys() else 'idx_{}'.format(batch['idx'])
                
            total_img_coarse, total_img_fine = [], []

            if opt.model in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                data_dict = self.evaluate_test_time_photometric_optim(opt, data_dict)
                # important is data_dict.pose_refine_test
            H, W = data_dict.image.shape[-2:]
            opt.H, opt.W = H, W

            output_dict = self.net.forward(opt,data_dict,mode="eval", iter=None)  

            # evaluate view synthesis, coarse
            scaling_factor_for_pred_depth = 1.
            if self.settings.model == 'joint_pose_nerf_training' and hasattr(self.net, 'sim3_est_to_gt_c2w'):
                # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
                # similarity, compared to the ground-truth. 
                scaling_factor_for_pred_depth = (self.net.sim3_est_to_gt_c2w.trans_scaling_after * self.net.sim3_est_to_gt_c2w.s) \
                    if self.net.sim3_est_to_gt_c2w.type == 'align_to_first' else self.net.sim3_est_to_gt_c2w.s

            # gt image
            gt_rgb_map = data_dict.image  # (B, 3, H, W)

            # rendered image and depth map
            pred_rgb_map = output_dict.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            pred_depth = output_dict.depth  # [B, -1, 1]
    
            results = compute_metrics(data_dict, output_dict, pred_rgb_map, pred_depth, gt_rgb_map, 
                                      lpips_loss=self.lpips_loss, 
                                      scaling_factor_for_pred_depth=scaling_factor_for_pred_depth, suffix='_c')

            if 'depth_fine' in output_dict.keys():
                pred_rgb_map_fine = output_dict.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                pred_depth_fine = output_dict.depth_fine
                results_fine = compute_metrics(data_dict, output_dict, pred_rgb_map_fine, 
                                               pred_depth_fine, gt_rgb_map, 
                                               scaling_factor_for_pred_depth=scaling_factor_for_pred_depth, 
                                               lpips_loss=self.lpips_loss, suffix='_f') 
                results.update(results_fine)       

            res.append(results)

            message = "==================\n"
            message += "{}, curr_id: {}, shape {}x{} \n".format(self.settings.scene, file_id, H, W)
            for k, v in results.items():
                message += 'current {}: {:.2f}\n'.format(k, v)
            self.logger.info(message)
            
            results_dict['single'][file_id] = results

            # plotting   
            depth_range = None
            if plot:
                # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                depth = output_dict.depth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) * scaling_factor_for_pred_depth # [B,1,H,W]
                depth_var = output_dict.depth_var.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                
                if hasattr(data_dict, 'depth_range'):
                    depth_range = data_dict.depth_range[0].cpu().numpy().tolist()

                self.visualize_eval(total_img_coarse, data_dict.image, pred_rgb_map, depth, 
                                    depth_var, depth_range=depth_range)
                if 'depth_gt' in data_dict.keys():
                    depth_gt = data_dict.depth_gt[0]
                    depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), 
                                                          range=depth_range, append_cbar=False)).astype(np.uint8)
                    total_img_coarse += [depth_gt_colored]

                if 'depth_fine' in output_dict.keys():
                    #invdepth_fine = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    #invdepth_map_fine = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    depth = output_dict.depth_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) * scaling_factor_for_pred_depth  # [B,1,H,W]
                    depth_var = output_dict.depth_var_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    self.visualize_eval(total_img_fine, data_dict.image, pred_rgb_map_fine, depth, 
                                        depth_var, depth_range=depth_range)
                    if 'depth_gt' in data_dict.keys():
                        depth_gt = data_dict.depth_gt[0]
                        depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), 
                                                            range=depth_range, append_cbar=False)).astype(np.uint8)
                        total_img_fine += [depth_gt_colored]
                
                # save the final image
                total_img_coarse = np.concatenate(total_img_coarse, axis=1)
                if len(total_img_fine) > 2:
                    total_img_fine = np.concatenate(total_img_fine, axis=1)
                    total_img = np.concatenate((total_img_coarse, total_img_fine), axis=0)
                else:
                    total_img = total_img_coarse
                if 'depth_gt' in data_dict.keys():
                    name = '{}_gt_rgb_depthc_depthvarc_depthgt.png'.format(file_id)
                else:
                    name = '{}_gt_rgb_depthc_depthvarc.png'.format(file_id)
                imageio.imwrite(os.path.join(out_scene_dir, name), total_img)  


            if save_ind_files: 
                pred_img_plot = pred_rgb_map_fine if 'depth_fine' in output_dict.keys() else pred_rgb_map
                pred_depth_plot = pred_depth_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) if 'depth_fine' in output_dict.keys() \
                    else pred_depth.view(-1,opt.H,opt.W,1).permute(0,3,1,2)
                
                depth_gt_image = data_dict.depth_gt if 'depth_gt' in data_dict.keys() else None
                self.save_ind_files(out_scene_dir, file_id, data_dict.image, pred_img_plot, 
                                    pred_depth_plot * scaling_factor_for_pred_depth, 
                                    depth_range=depth_range, depth_gt=depth_gt_image)

        # compute average results over the full test set
        avg_results = {}
        keys = res[0].keys()
        for key in keys:
            avg_results[key] = np.mean([r[key] for r in res])
        results_dict['mean'] = avg_results
        
        # log results
        message = "------avg over {}-------\n".format(self.settings.scene)
        for k, v in avg_results.items():
            message += 'current {}: {:.3f}\n'.format(k, v)
        self.logger.info(message)
        return results_dict