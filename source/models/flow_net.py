import os
import sys
import math
import gc
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image

sys.path.append(str(Path(__file__).parent / '../../third_party/DenseMatching'))
sys.path.append(str(Path(__file__).parent / '../../third_party/roma'))

from third_party.DenseMatching.utils_flow.pixel_wise_mapping import warp
from third_party.DenseMatching.utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from third_party.DenseMatching.models.PDCNet.base_pdcnet import estimate_probability_of_confidence_interval_of_mixture_density, estimate_average_variance_of_mixture_density
from third_party.DenseMatching.utils_flow.pixel_wise_mapping import remap_using_correspondence_map



class FlowSelectionWrapper(nn.Module):
    """Wrapper for flow networks to compute flows/correspondence maps between image pairs.
    Designed for processing a few input views - many views may cause memory issues."""

    def __init__(self, ckpt_path, num_views, backbone='PDCNet', batch_size=5):
        """Initialize flow selection wrapper.
        
        Args:
            ckpt_path: Path to correspondence network checkpoint
            num_views: Number of training views
            backbone: Network backbone type ('PDCNet', 'SPSG', 'RoMa_indoor', 'RoMa_outdoor')
            batch_size: Batch size for computing correspondence
        """
        super().__init__()
    
        self.backbone = backbone
        self.confidence_map_type = 'p_r'

        # GTFlowFromProjection Does not need network
        if backbone is not 'GTFlowFromProjection':
            self.load_flow_network(backbone, ckpt_path=ckpt_path)

        self.batch_size = batch_size
        self.num_views = num_views
        self.combi_list = get_combi_list(
            num_views,
            method='all') 
        # shape is 2x(num_views*(num_views - 1))
        # 2x(num_views*(num_views - 1)). [[0, 0, 0, ... 1, 1, 1], [1, 2, 3, ..9, 0, 2, 3, 4, 5, ]]
        # all combinations except for oneself

    def load_flow_network(self, backbone, ckpt_path=None):
        """Initialize and load flow network with optional checkpoint.
        
        Args:
            backbone: Network architecture type
            ckpt_path: Optional path to model checkpoint
        """
        self.flow_net = flow_net_model_select(backbone)

        if ckpt_path is not None:
            self.flow_net = self.load_network(backbone, ckpt_path)

        # Freeze network weights
        self.flow_net.requires_grad_ = False
        if self.backbone != 'SPSG':
            for param in self.flow_net.parameters():
                param.requires_grad = False
        
        self.flow_net.eval()

    def load_network(self, backbone, checkpoint_path):
        """Load network weights from checkpoint.
        
        Args:
            backbone: Network architecture type
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded network with weights from checkpoint
        """
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f'Checkpoint does not exist: {checkpoint_path}')

        if hasattr(self.flow_net, 'load_weights'):
            self.flow_net.load_weights(checkpoint_path)
        else:
            print(f'Loading flow checkpoint from {checkpoint_path}')
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint_dict:
                self.flow_net.load_state_dict(checkpoint_dict['state_dict'])
            else:
                self.flow_net.load_state_dict(checkpoint_dict)

        return self.flow_net


    def compute_flow_and_confidence_map_of_combi_list(
            self, images, combi_list_tar_src, plot=False, use_homography=False, additional_data=None
    ):
        '''Computing flow and confidence map of set of images given combi_list.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot,
                additional_data: Additional Data for GT Correspondence from Projection
        '''
        if self.backbone == 'SPSG':
            return self.compute_matches_spsg(images, combi_list_tar_src, plot=plot)
        elif self.backbone == 'PDCNet':
            return self.compute_matches_pdcnet(images, combi_list_tar_src, plot, use_homography)
        elif self.backbone == 'GTFlowFromProjection':
            use_homography = False
            return self.compute_matches_gtprojection(images, combi_list_tar_src, plot, use_homography=use_homography, additional_data=additional_data)
        elif self.backbone == 'RoMa_indoor' or self.backbone == 'RoMa_outdoor':
            return self.compute_matches_roma(images, combi_list_tar_src, plot, use_homography)

    def compute_matches_gtprojection(
            self, images, combi_list_tar_src, plot=False, use_homography=False, additional_data=None
    ):
        '''Computing flow and confidence map of set of images given combi_list, using PDC-Net.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                mapping_self_to_neighbor (torch.Tensor): correspondence map from target to source,
                                                          shape [len(combi_list), 2/1, H, W]
                confidence_map (torch.Tensor): correspondence map from target to source,
                                               shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot,
                additional_data: Additional Intrinsic, Extrinsic and Depthmap for Projection
        '''
        from source.utils.camera import pad_poses, unpad_poses
        import einops, kornia, cv2
        n_views, _, H, W = images.shape

        # set deterministic combination if not chosen online
        if combi_list_tar_src is None:
            combi_list_tar_src = self.combi_list

        mapping_self_to_neighbor, confidence_map = list(), list()
        for ind in range(combi_list_tar_src.shape[1]):
            source = combi_list_tar_src[1, ind]
            target = combi_list_tar_src[0, ind]
            depth_gt = additional_data['depth_gt'][target].view([1, 1, H, W])
            depth_gt_source = additional_data['depth_gt'][source].view([1, 1, H, W])
            intr = additional_data['intr'][source]
            intr44 = torch.eye(4).cuda()
            intr44[0:3, 0:3] = intr
            # World to Camera Pose
            pose_source, pose_target = pad_poses(additional_data['pose'][source]), pad_poses(additional_data['pose'][target])
            prjmatrix = intr @ unpad_poses(pose_source @ pose_target.inverse()) @ intr44.inverse()
            prjmatrix = prjmatrix.view([1, 1, 1, 3, 4])

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()
            yy = yy.view(1, 1, H, W).repeat(1, 1, 1, 1).float().cuda()

            pts3d = torch.cat([xx * depth_gt, yy * depth_gt, depth_gt, torch.ones_like(depth_gt)], dim=1)
            pts3d = einops.rearrange(pts3d, 'b n h w -> b h w n 1')
            ptsprj = prjmatrix @ pts3d
            ptsprjx, ptsprjy = ptsprj[:, :, :, 0, 0] / (ptsprj[:, :, :, 2, 0] + 1e-10), ptsprj[:, :, :, 1, 0] / (ptsprj[:, :, :, 2, 0] + 1e-10)

            ptsprjx_ = (ptsprjx / W - 0.5) * 2
            ptsprjy_ = (ptsprjy / H - 0.5) * 2
            ptsprjx_grid = torch.stack([ptsprjx_, ptsprjy_], dim=-1)
            depth_recon = torch.nn.functional.grid_sample(depth_gt_source, ptsprjx_grid, mode='bilinear', align_corners=False)
            depth_consis = (depth_recon - ptsprj[:, None, :, :, 2, 0]).abs()
            '''
            from analysis.utils_vls import tensor2rgb, tensor2disp
            import PIL.Image as Image
            tensor2disp(1 / depth_gt, vmax=1.0, viewind=0).show()
            tensor2disp(1 / depth_recon, vmax=1.0, viewind=0).show()
            tensor2disp(depth_consis < 0.05, vmax=0.1, viewind=0).show()
            '''
            confidence_map_ = (ptsprjx > 0) * (ptsprjy > 0) * (ptsprjx < W - 1) * (ptsprjy < H - 1) * (depth_consis < 0.05)
            confidence_map_ = confidence_map_.unsqueeze(1) * (depth_gt > 0)
            ptsprjx[(confidence_map_ == 0).squeeze(1).squeeze(1)] = torch.nan
            ptsprjy[(confidence_map_ == 0).squeeze(1).squeeze(1)] = torch.nan

            confidence_map.append(confidence_map_.float())
            mapping_self_to_neighbor.append(torch.stack([ptsprjx, ptsprjy], dim=1))

            '''
            from analysis.utils_vls import tensor2rgb, tensor2disp
            import PIL.Image as Image
            vls_source = tensor2rgb(additional_data['image'], viewind=source)
            vls_target = tensor2rgb(additional_data['image'], viewind=target)
            vls_depth = tensor2disp(1 / (depth_gt + 1e-3), viewind=0, vmax=0.5)

            ptsprjx_ = (ptsprjx / W - 0.5) * 2
            ptsprjy_ = (ptsprjy / H - 0.5) * 2
            ptsprjx_grid = torch.stack([ptsprjx_, ptsprjy_], dim=-1)
            vls_recon = torch.nn.functional.grid_sample(additional_data['image'][source:source+1], ptsprjx_grid, mode='bilinear')
            vls_recon = tensor2rgb(vls_recon, viewind=0)
            vls_combined = np.concatenate([np.array(vls_source), np.array(vls_target), np.array(vls_depth), np.array(vls_recon)], axis=0)
            Image.fromarray(vls_combined).show()
            '''

        confidence_map = torch.cat(confidence_map, dim=0).contiguous()
        mapping_self_to_neighbor = torch.cat(mapping_self_to_neighbor, dim=0).contiguous()

        ret = [mapping_self_to_neighbor, confidence_map]  # (B, 2, H, W) and (B, 1, H, W)

        if plot:
            flow_plot = None
            if confidence_map.shape[0] < 1000:
                flow_plot = self.visualize_mapping_combinations(images, mapping_self_to_neighbor, confidence_map, combi_list_tar_src, save_path=None)
                flow_plot = torch.from_numpy(flow_plot.astype(np.float32) / 255.).permute(2, 0, 1)

            ret += [flow_plot]

        return ret

    def compute_flow_and_confidence_map_and_cc_of_combi_list(
            self, images, combi_list_tar_src, plot=False, use_homography=False
    ):
        '''Computing flow and confidence map of set of images given combi_list. Apply cyclic consistency 
        as an additional filtering mechanism for the matches. 
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''
        if self.backbone == 'SPSG':
            return self.compute_matches_spsg(images, combi_list_tar_src, plot=plot, return_dummy_cc_map=True)
        else:
            return self.compute_matches_pdcnet_with_cc(images, combi_list_tar_src, plot, use_homography)


    # ---------------------------- SPSG matches --------------------------------
    def compute_matches_spsg(self, images, combi_list_tar_src, plot=False, return_dummy_cc_map=False):
        '''Computing flow and confidence map of set of images given combi_list, using SuperPoint and SuperGlue.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''
        plot_ = plot and combi_list_tar_src.shape[1] < 100

        B, _, H, W = images.shape


        images_proc = self.flow_net.pre_process_img(images * 255)
        batch_size = 50

        # extract keypoints for all images
        kp_dict =  {}  
        # kp_dict will contain 'keypoint', 'scores'
        # in each, there is a list of lists, i.e. an element per image
        for idx_start in range(0, images.shape[0], batch_size):
            if idx_start == images.shape[0] - 1:
                kp_template_dict_ = self.flow_net.get_keypoints\
                    (images_proc[idx_start].unsqueeze(0))
            else:
                kp_template_dict_ = self.flow_net.get_keypoints\
                    (images_proc[idx_start:idx_start+batch_size])

            for k, v in kp_template_dict_.items():
                if k in kp_dict.keys():
                    kp_dict[k].extend(kp_template_dict_[k])
                else:
                    kp_dict[k] = kp_template_dict_[k]
            torch.cuda.empty_cache()

        correspondence_map = torch.zeros(combi_list_tar_src.shape[1], H, W, 2).to(images.device)
        conf_map = torch.zeros(combi_list_tar_src.shape[1], H, W, 1).to(images.device)

        # establish matches
        plot_list = [] if plot_ else None
        for idx in range(combi_list_tar_src.shape[1]):  # (2xN)
            id_target, id_source = combi_list_tar_src[:, idx] 
            source_kp_dict = {k: [v[id_source]] for k, v in kp_dict.items()}
            target_kp_dict = {k: [v[id_target]] for k, v in kp_dict.items()}
            pred = self.flow_net.get_matches_and_confidence(source_img=images_proc[id_source].unsqueeze(0), 
                                                            target_img=images_proc[id_target].unsqueeze(0), 
                                                            source_kp_dict=source_kp_dict, 
                                                            target_kp_dict=target_kp_dict, 
                                                            preprocess_image=False)
            # 'kp_source': mkpts0, 'kp_target': mkpts1, 'confidence_value': mconf
            pred_kp_target = torch.from_numpy(pred['kp_target']).to(images.device)  # Nx2
            diff = torch.round(pred_kp_target) - pred_kp_target
            pred_kp_target = torch.round(pred_kp_target).long()
            pred_kp_source = torch.from_numpy(pred['kp_source']).to(images.device) + diff  # Nx2
            
            if plot_:
                plot_list.append(make_matching_plot_fast(
                    image1=(images[id_source].permute(1, 2, 0).cpu().detach().numpy() * 255), 
                    image0=(images[id_target].permute(1, 2, 0).cpu().detach().numpy() * 255), 
                    kpts1=pred['kp_source'][:500], kpts0=pred['kp_target'][:500], 
                    text=['{} to {}'.format(id_source, id_target), 
                          '{} matches'.format(pred_kp_source.shape[0]), 'Top 500 matches']))
            pred_conf = torch.from_numpy(pred['confidence_value']).to(images.device)
            
            correspondence_map[idx, pred_kp_target[:, 1], pred_kp_target[:, 0]] = pred_kp_source
            conf_map[idx, pred_kp_target[:, 1], pred_kp_target[:, 0]] = pred_conf.reshape(-1, 1)
        
        ret = [correspondence_map.permute(0, 3, 1, 2), conf_map.permute(0, 3, 1, 2)]
        if return_dummy_cc_map:
            ret += [torch.ones_like(conf_map.permute(0, 3, 1, 2))]

        if plot:
            if plot_list is not None:
                plot_list = np.concatenate(plot_list, axis=0)
                plot_list = torch.from_numpy(plot_list.astype(np.float32)).permute(2, 0, 1)
            ret += [plot_list]
        return ret

    # --------------------- main function to compute pdcnet matches ------------------------
    def compute_matches_pdcnet(self, images, combi_list_tar_src, plot=False, use_homography=False):
        """Compute flow and confidence maps between image pairs using PDC-Net.
        
        Args:
            images: Normalized image tensor [B, C, H, W]
            combi_list_tar_src: Image pair indices [2, N], first row is target indices
            plot: Whether to generate visualization plots
            use_homography: Whether to use homography estimation
            
        Returns:
            List containing:
            - mapping_self_to_neighbor: Correspondence map from target to source [N, 2, H, W]
            - confidence_map: Confidence values for correspondences [N, 1, H, W]
            - flow_plot: Optional visualization tensor if plot=True
        """

        n_views, _, H, W = images.shape
        img_size = (H, W)

        # Use default combinations if none provided
        if combi_list_tar_src is None:
            combi_list_tar_src = self.combi_list

        # Scale images to [0, 255] range
        images = images * 255.

        # Process images for flow computation
        flow_data = images if use_homography else self.process_data_for_flow_net(images, extract_features=False)

        # Compute forward flow and confidence
        return_confidence_map = self.confidence_map_type != 'cyclic_consistency_error'
        flow_self_to_neighbor, confidence_map = self.compute_flow_combinations(
            flow_data, 
            torch.flip(combi_list_tar_src, [0]),  # First element becomes source
            img_size, 
            return_confidence_map=return_confidence_map,
            use_homography=use_homography
        )

        # Handle cyclic consistency confidence
        if self.confidence_map_type == 'cyclic_consistency_error':
            # Compute backward flow
            flow_neighbor_to_self = self.compute_flow_combinations(
                flow_data, combi_list_tar_src, img_size, use_homography=use_homography
            )[0]

            # Compute cyclic consistency error
            cyclic_consistency_error = flow_self_to_neighbor + warp(flow_neighbor_to_self, flow_self_to_neighbor)
            cyclic_consistency_error = torch.norm(cyclic_consistency_error, dim=1, keepdim=True)
            confidence_map = 1.0 / (1.0 + cyclic_consistency_error)
        elif confidence_map is None:
            raise ValueError("Confidence map computation failed")

        # Convert flow to mapping
        mapping_self_to_neighbor = convert_flow_to_mapping(flow_self_to_neighbor, output_channel_first=True)

        # Prepare return values
        ret = [mapping_self_to_neighbor, confidence_map]

        # Generate visualization if requested
        if plot and confidence_map.shape[0] < 1000:
            flow_plot = self.visualize_mapping_combinations(
                images / 255., 
                mapping_self_to_neighbor,
                confidence_map, 
                combi_list_tar_src,
                save_path=None,
            )
            flow_plot = torch.from_numpy(flow_plot.astype(np.float32) / 255.).permute(2, 0, 1)
            ret.append(flow_plot)

        return ret


    def compute_matches_pdcnet_with_cc(self, images, combi_list, plot=False, use_homography=False):
        '''Computing flow and confidence map of set of images using given combi_list.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                mapping_self_to_neighbor (torch.Tensor): correspondence map from target to source, 
                                                          shape [len(combi_list), 2/1, H, W]
                confidence_map (torch.Tensor): correspondence map from target to source, 
                                               shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''

        n_views, _, H, W = images.shape
        img_size = (H, W)

        # set deterministic combination if not chosen online
        if combi_list is None:
            combi_list = self.combi_list

        images = images * 255.

        extract_features = False
        if use_homography:
            flow_data = images
        else:
            flow_data = self.process_data_for_flow_net(images, extract_features=extract_features)

        # can query projection points in self
        return_confidence_map = True
        flow_self_to_neighbor, confidence_map = self.compute_flow_combinations\
            (flow_data, torch.flip(combi_list, [0]),  # the first element will now be the target
             img_size, return_confidence_map=return_confidence_map, use_homography=use_homography)  # (B, 2, H, W)
        confidence_map = confidence_map  # (B, 1, H, W)

        flow_neighbor_to_self, conf_map_neighbor_to_self = self.compute_flow_combinations\
            (flow_data, combi_list, img_size, use_homography=use_homography)

        gc.collect()
        with torch.no_grad():
            # we want consistency error in self coordinates
            if flow_neighbor_to_self.shape[0] > 500:
                inter = 100
                cyclic_consistency_error = []
                for i_start in range(0, flow_neighbor_to_self.shape[0], inter):
                    cyclic_consistency_error_ = flow_self_to_neighbor[i_start: i_start+inter] + \
                        warp(flow_neighbor_to_self[i_start: i_start+inter], flow_self_to_neighbor[i_start: i_start+inter])
                    cyclic_consistency_error.append(cyclic_consistency_error_.cpu())
                    torch.cuda.empty_cache()

                cyclic_consistency_error = torch.cat(cyclic_consistency_error, dim=0) if len(cyclic_consistency_error) > 1\
                    else cyclic_consistency_error[0]
                cyclic_consistency_error = cyclic_consistency_error.to(confidence_map.device)
            else:
                cyclic_consistency_error = flow_self_to_neighbor + warp(flow_neighbor_to_self, flow_self_to_neighbor)
            cyclic_consistency_error = torch.norm(cyclic_consistency_error, dim=1, keepdim=True)  # (B, 1, H, W)
            confidence_map_from_cc = 1.0 / (1.0 + cyclic_consistency_error) 

        ret = []
        # return_correspondence_map:
        mapping_self_to_neighbor = convert_flow_to_mapping(flow_self_to_neighbor, output_channel_first=True)  # (B, 2, H, W)
        ret += [mapping_self_to_neighbor, confidence_map, confidence_map_from_cc]

        if plot:
            flow_plot = None
            if confidence_map.shape[0] < 1000:
                flow_plot = self.visualize_mapping_combinations(images / 255., mapping_self_to_neighbor, 
                                                                confidence_map, combi_list, save_path=None)
                flow_plot = torch.from_numpy( flow_plot.astype(np.float32)/255.).permute(2, 0, 1)
            
            ret += [flow_plot]
        # all the correspondences stuff are (B, 2/1, H, W)
        return ret

    # --------------------- main function to compute pdcnet matches ------------------------
    def compute_matches_roma(self, images, combi_list_tar_src, plot=False, use_homography=False):
        """Compute flow and confidence maps between image pairs using RoMa network.
        
        Args:
            images: Normalized image tensor [B, C, H, W]
            combi_list_tar_src: Image pair indices [2, N], first row is target indices
            plot: Whether to generate visualization plots
            use_homography: Whether to use homography estimation (unused in RoMa)
            
        Returns:
            List containing:
            - mapping_self_to_neighbor: Correspondence map from target to source [N, 2, H, W]
            - confidence_map: Confidence values for correspondences [N, 1, H, W]
            - flow_plot: Optional visualization tensor if plot=True
        """

        n_views, _, H, W = images.shape

        # Use default combinations if none provided
        if combi_list_tar_src is None:
            combi_list_tar_src = self.combi_list

        # Process each image pair
        warps, certaintys = [], []
        for i in range(combi_list_tar_src.shape[1]):
            # Get source and target image indices
            src_idx, dst_idx = combi_list_tar_src[0, i].item(), combi_list_tar_src[1, i].item()
            
            # Convert tensors to PIL images
            img_src = Image.fromarray(
                (images[src_idx].view([3, H, W]) * 255.0).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
            )
            img_dst = Image.fromarray(
                (images[dst_idx].view([3, H, W]) * 255.0).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
            )
            
            # Compute correspondence and certainty
            warp, certainty = self.flow_net.match(img_src, img_dst, device=torch.device("cuda"))
            warp = warp[:, :, 2:4]  # Extract relevant warp dimensions
            warps.append(warp)
            certaintys.append(certainty.view([560, 560, 1]))

        # Stack and process warps and certainties
        warps = torch.stack(warps, dim=0)
        certaintys = torch.stack(certaintys, dim=0)
        
        # Mask out invalid warp coordinates
        warpsx, warpsy = torch.split(warps, 1, dim=-1)
        valid_mask = (warpsx > -1) * (warpsx < 1) * (warpsy > -1) * (warpsy < 1)
        certaintys = certaintys * valid_mask

        # Combine warps and certainties
        corres_confidence = torch.cat([warps, certaintys], dim=-1).permute([0, 3, 1, 2])
        
        # Resize to match input dimensions
        corres_confidence = F.interpolate(
            corres_confidence, [H, W], 
            align_corners=True, 
            mode='bilinear'
        )

        # Split and process correspondence maps
        mapping_self_to_neighbor, confidence_map = torch.split(corres_confidence, [2, 1], dim=1)
        
        # Convert normalized coordinates to pixel coordinates
        mapping_self_to_neighborx, mapping_self_to_neighbory = torch.split(mapping_self_to_neighbor, 1, dim=1)
        mapping_self_to_neighborx = (mapping_self_to_neighborx + 1) / 2 * W - 0.5
        mapping_self_to_neighbory = (mapping_self_to_neighbory + 1) / 2 * H - 0.5
        mapping_self_to_neighbor = torch.cat([mapping_self_to_neighborx, mapping_self_to_neighbory], dim=1)

        # Prepare return values
        ret = [mapping_self_to_neighbor, confidence_map]

        # Generate visualization if requested
        if plot and confidence_map.shape[0] < 1000:
            flow_plot = self.visualize_mapping_combinations(
                images, 
                mapping_self_to_neighbor,
                confidence_map, 
                combi_list_tar_src,
                save_path=None
            )
            flow_plot = torch.from_numpy(flow_plot.astype(np.float32) / 255.).permute(2, 0, 1)
            ret.append(flow_plot)

        return ret

    # ------------- functions for processing and computing matches for PDCNet --------------------
        # ------------------------------- PDCNet matches ------------------------------------
    @staticmethod
    @torch.no_grad()
    def pre_process_imgs(imgs, mean_vector=[0.485, 0.456, 0.406], std_vector=[0.229, 0.224, 0.225]):
        """Preprocess images for PDC-Net inference.
        
        Performs the following operations:
        1. Resizes images to dimensions divisible by 8 (or 256 if smaller)
        2. Normalizes using ImageNet mean and std
        3. Creates a 256x256 version for multi-scale processing
        
        Args:
            imgs: Input images [B, C, H, W] in uint8 format
            mean_vector: ImageNet RGB mean values for normalization
            std_vector: ImageNet RGB std values for normalization
            
        Returns:
            Tuple containing:
            - imgs_: Resized and normalized images [B, C, H', W']
            - imgs_256: 256x256 version of normalized images
            - scale_x: Width scaling factor between original and resized
            - scale_y: Height scaling factor between original and resized
        """

        # Get input dimensions
        _, _, H, W = imgs.shape

        # Calculate dimensions divisible by 8 (or 256 if smaller)
        H_int = int(math.floor(int(H / 8.0) * 8.0)) if H > 256 else 256
        W_int = int(math.floor(int(W / 8.0) * 8.0)) if W > 256 else 256

        # Convert normalization params to tensors
        mean = torch.as_tensor(mean_vector, dtype=torch.float32, device=imgs.device)
        std = torch.as_tensor(std_vector, dtype=torch.float32, device=imgs.device)

        # Resize and normalize main images
        imgs_ = F.interpolate(
            input=imgs.float(),
            size=(H_int, W_int),
            mode='area'
        ).byte().float().div(255.)
        imgs_.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Create and normalize 256x256 version
        imgs_256 = F.interpolate(
            input=imgs.float(),
            size=(256, 256),
            mode='area'
        ).byte().float().div(255.)
        imgs_256.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Calculate scaling factors
        scale_x = float(W) / float(W_int)
        scale_y = float(H) / float(H_int)

        return imgs_, imgs_256, scale_x, scale_y

    
    def process_data_for_flow_net(self, imgs, extract_features=True):
        """Process images for flow network inference.
        
        Performs preprocessing and optionally extracts feature pyramids:
        1. Normalizes and resizes images using pre_process_imgs
        2. Optionally extracts multi-scale feature pyramids
        3. Handles batching for memory efficiency
        
        Args:
            imgs: Input images [B, C, H, W]
            extract_features: Whether to extract feature pyramids
            
        Returns:
            Tuple containing:
            - imgs: Preprocessed full resolution images
            - imgs_256: Preprocessed 256x256 images
            - imgs_pyr: Feature pyramids for full res (if extract_features=True)
            - imgs_pyr_256: Feature pyramids for 256x256 (if extract_features=True)
            - scale_x: Width scaling factor
            - scale_y: Height scaling factor
        """
        # Preprocess images
        with torch.no_grad():
            imgs, imgs_256, scale_x, scale_y = self.pre_process_imgs(imgs)

        # Initialize feature pyramids
        imgs_pyr, imgs_pyr_256 = None, None

        # Extract feature pyramids if requested
        if extract_features:
            imgs_pyr, imgs_pyr_256 = [], []
            batch_size = 500  # Process in batches to manage memory

            # Extract features in batches
            for i_start in range(0, imgs.shape[0], batch_size):
                # Get batch slice
                batch_end = min(i_start + batch_size, imgs.shape[0])
                imgs_batch = imgs[i_start:batch_end]
                imgs_256_batch = imgs_256[i_start:batch_end]

                # Extract pyramids for batch
                imgs_pyr_, imgs_pyr_256_ = self.flow_net.extract_pyramid(
                    imgs_batch, imgs_256_batch
                )
                imgs_pyr.append(imgs_pyr_)
                imgs_pyr_256.append(imgs_pyr_256_)
                
                torch.cuda.empty_cache()

            # Concatenate batched pyramids
            imgs_pyr = torch.cat(imgs_pyr, dim=0) if len(imgs_pyr) > 1 else imgs_pyr[0]
            imgs_pyr_256 = torch.cat(imgs_pyr_256, dim=0) if len(imgs_pyr_256) > 1 else imgs_pyr_256[0]

        return imgs, imgs_256, imgs_pyr, imgs_pyr_256, scale_x, scale_y

    def compute_flow_combinations(self, flow_data, combi_list, output_shape, return_confidence_map=False, use_homography=False):
        """Compute optical flow and confidence maps for multiple image pairs.
        
        Processes image pairs in batches to compute:
        1. Optical flow fields between source and target images
        2. Optional confidence maps for the computed flows
        3. Optional homography-based flow estimation
        
        Args:
            flow_data: Preprocessed image data from process_data_for_flow_net
            combi_list: Image pair indices [2, N], first row is source indices
            output_shape: Target output resolution (H, W)
            return_confidence_map: Whether to compute confidence maps
            use_homography: Whether to use homography estimation
            
        Returns:
            Tuple containing:
            - flow_est: Optical flow fields [N, 2, H, W]
            - batched_conf_map: Confidence maps [N, 1, H, W] if requested
        """

        def output_to_flow_and_uncertainty(output):
            """Extract flow and uncertainty estimates from network output.
            
            Args:
                output: Network output dictionary containing flow and uncertainty estimates
                
            Returns:
                Tuple of:
                - flow_est: Final flow field estimate
                - p_r: Confidence map (None if no uncertainty estimates)
            """
            flow_est = output['flow_estimates'][-1]  # Use final flow estimate
            p_r = None
            
            if 'uncertainty_estimates' in output:
                # Extract uncertainty components
                log_var_map, weight_map = output['uncertainty_estimates'][-1]
                
                # Compute confidence value
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(
                    weight_map, log_var_map, R=1.
                ) / 0.5730
                
            return flow_est, p_r


        # Initialize output containers
        batched_flow, batched_conf_map = [], []

        if use_homography:
            # Use homography-based flow estimation
            imgs = flow_data  # Images in [0, 255] range
            
            # Process each image pair
            for idx in range(combi_list.shape[1]):
                # Extract source and target images
                src_imgs = imgs[combi_list[0, idx]].unsqueeze(0)
                tgt_imgs = imgs[combi_list[1, idx]].unsqueeze(0)

                # Compute flow and uncertainty
                estimated_flow, uncertainty_dict = self.flow_net.estimate_flow_and_confidence_map_with_homo(
                    src_imgs, tgt_imgs,
                    inference_parameters=self.flow_net.inference_parameters,
                    scaling=1.0/4.,
                    mode='channel_first'
                )

                # Store results
                batched_flow.append(estimated_flow)
                if return_confidence_map:
                    conf_map = uncertainty_dict['p_r'] / 0.5730
                    batched_conf_map.append(conf_map)

                torch.cuda.empty_cache()

            # Concatenate results
            flow_est = torch.cat(batched_flow, dim=0)
            if return_confidence_map:
                batched_conf_map = torch.cat(batched_conf_map, dim=0)
            else:
                batched_conf_map = None
        else:
            batched_combi_list = torch.split(combi_list, self.batch_size, dim=1)  # list of elements of size [2, self.batch_size]
            imgs, imgs_256, imgs_pyr, imgs_pyr_256, scale_x, scale_y = flow_data

            for idx in batched_combi_list:
                # batched_combi_list shape [2, self.batch_size], first element is source, second is target
                src_imgs = imgs[idx[0], ...]
                tgt_imgs = imgs[idx[1], ...]

                src_imgs_256 = imgs_256[idx[0], ...]
                tgt_imgs_256 = imgs_256[idx[1], ...]

                if imgs_pyr is not None:
                    src_imgs_pyr = [pyr[idx[0], ...] for pyr in imgs_pyr]
                    tgt_imgs_pyr = [pyr[idx[1], ...] for pyr in imgs_pyr]

                    src_imgs_pyr_256 = [pyr[idx[0], ...] for pyr in imgs_pyr_256]
                    tgt_imgs_pyr_256 = [pyr[idx[1], ...] for pyr in imgs_pyr_256]

                    # batch process this
                    _ , output = self.flow_net.forward(
                        tgt_imgs, src_imgs,
                        tgt_imgs_256, src_imgs_256,
                        im_target_pyr=tgt_imgs_pyr,
                        im_source_pyr=src_imgs_pyr,
                        im_target_pyr_256=tgt_imgs_pyr_256,
                        im_source_pyr_256=src_imgs_pyr_256
                    )
                else:
                    _ , output = self.flow_net.forward(
                        tgt_imgs, src_imgs,
                        tgt_imgs_256, src_imgs_256
                    )


                flow_est, conf_map = output_to_flow_and_uncertainty(output)
                flow_est = F.interpolate(input=flow_est,
                                        size=output_shape,
                                        mode='bilinear',
                                        align_corners=False)
                batched_flow.append(flow_est)
                if conf_map is not None and return_confidence_map:
                    conf_map = F.interpolate(input=conf_map, size=output_shape, mode='bilinear',
                                             align_corners=False)
                    batched_conf_map.append(conf_map)

                torch.cuda.empty_cache() # otherwise GPU memory filles up quickly


            flow_est = torch.cat(batched_flow, dim=0)
            flow_est[:, 0, :, :] *= scale_x
            flow_est[:, 1, :, :] *= scale_y
            if len(batched_conf_map) > 0 and return_confidence_map:
                batched_conf_map = torch.cat(batched_conf_map, dim=0)
            else:
                batched_conf_map = None

        return flow_est, batched_conf_map

    @torch.no_grad()
    def visualize_mapping_combinations(self, images, mapping_est, batched_conf_map, combi_list, save_path):
        return visualize_mapping_combinations(images, mapping_est, batched_conf_map, combi_list, save_path)
    
    # ------------- on image pair -----------------------------
    def pair_flow_forward(self, src_img, target_img, return_correspondence_map=False):
        '''
        for an image pair only, computes the flow field relating the target to the source image. 
        src_img: BxCxHxW normalized to [0, 1]
        params:target_img:'
        '''
        H, W = target_img.shape[-2:]
        img_size = (H, W)

        flow_data = self.process_data_for_flow_net(src_img  * 255. )
        flow_data_tgt = self.process_data_for_flow_net(target_img  * 255.)

        src_imgs, src_imgs_256, src_imgs_pyr, src_imgs_pyr_256, scale_x, scale_y = flow_data
        tgt_imgs, tgt_imgs_256, tgt_imgs_pyr, tgt_imgs_pyr_256, scale_x, scale_y = flow_data_tgt

        def output_to_flow_and_uncertainty(output):
            # for pdcnet
            p_r = None
            flow_est_list = output['flow_estimates']
            flow_est = flow_est_list[-1]
            if 'uncertainty_estimates' in output.keys():
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the confidence value
                log_var_map = uncertainty_list[0]
                weight_map = uncertainty_list[1]
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.) / 0.5730
            return flow_est, p_r

        _ , output = self.flow_net.forward(
            tgt_imgs, src_imgs,
            tgt_imgs_256, src_imgs_256,
            im_target_pyr=tgt_imgs_pyr,
            im_source_pyr=src_imgs_pyr,
            im_target_pyr_256=tgt_imgs_pyr_256,
            im_source_pyr_256=src_imgs_pyr_256
        )

        flow_est, p_r = output_to_flow_and_uncertainty(output)

        flow_est = F.interpolate(input=flow_est,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)

        flow_est[:, 0, :, :] *= scale_x
        flow_est[:, 1, :, :] *= scale_y
        if p_r is not None:
            p_r = F.interpolate(input=p_r,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)
        if return_correspondence_map:
            mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)  # (B, 2, H, W)
            return mapping, p_r
        return flow_est, p_r

    def pair_flow_forward_w_uncertainty(self, src_img, target_img, return_correspondence_map=False, return_conf_from_cc=False):
        '''
        for an image pair only, computes the flow field relating the target to the source image. 
        src_img: BxCxHxW normalized to [0, 1]
        params:target_img:'
        '''
        H, W = target_img.shape[-2:]
        img_size = (H, W)

        flow_data = self.process_data_for_flow_net(src_img  * 255. )
        flow_data_tgt = self.process_data_for_flow_net(target_img  * 255.)

        src_imgs, src_imgs_256, src_imgs_pyr, src_imgs_pyr_256, scale_x, scale_y = flow_data
        tgt_imgs, tgt_imgs_256, tgt_imgs_pyr, tgt_imgs_pyr_256, scale_x, scale_y = flow_data_tgt

        def output_to_flow_and_uncertainty(output):
            # for pdcnet
            p_r = None
            flow_est_list = output['flow_estimates']
            flow_est = flow_est_list[-1]
            if 'uncertainty_estimates' in output.keys():
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the confidence value
                log_var_map = uncertainty_list[0]
                weight_map = uncertainty_list[1]
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.) / 0.5730
                var = estimate_average_variance_of_mixture_density(weight_map, log_var_map)
            return flow_est, p_r, var

        _ , output = self.flow_net.forward(
            tgt_imgs, src_imgs,
            tgt_imgs_256, src_imgs_256,
            im_target_pyr=tgt_imgs_pyr,
            im_source_pyr=src_imgs_pyr,
            im_target_pyr_256=tgt_imgs_pyr_256,
            im_source_pyr_256=src_imgs_pyr_256
        )

        flow_est, p_r, var = output_to_flow_and_uncertainty(output)

        flow_est = F.interpolate(input=flow_est,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)

        flow_est[:, 0, :, :] *= scale_x
        flow_est[:, 1, :, :] *= scale_y
        p_r = F.interpolate(input=p_r,size=img_size,mode='bilinear', align_corners=False)
        var = F.interpolate(input=var,size=img_size,mode='bilinear', align_corners=False)

        ret = []
        if return_correspondence_map:
            mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)  # (B, 2, H, W)
            ret += [mapping, p_r, var]
        else:
            ret += [flow_est, p_r, var]

        if return_conf_from_cc:
            _ , output_src_to_tar = self.flow_net.forward(
                src_imgs, tgt_imgs, 
                src_imgs_256, tgt_imgs_256, 
                im_target_pyr=src_imgs_pyr,
                im_source_pyr=tgt_imgs_pyr,
                im_target_pyr_256=src_imgs_pyr_256,
                im_source_pyr_256=tgt_imgs_pyr_256
            )
            flow_est_src_to_tar, p_r_src_to_tar, var_src_to_tar = output_to_flow_and_uncertainty(output_src_to_tar)

            flow_est_src_to_tar = F.interpolate(input=flow_est_src_to_tar,
                                    size=img_size,
                                    mode='bilinear',
                                    align_corners=False)

            flow_est_src_to_tar[:, 0, :, :] *= scale_x
            flow_est_src_to_tar[:, 1, :, :] *= scale_y

            consistency_error = flow_est + warp(flow_est_src_to_tar, flow_est)  #flow_neighbor_to_self was created by just exchanging the source and the target basically 
            conf_from_consistency_error = 1. / (1 + torch.norm(consistency_error, dim=1, keepdim=True))
            ret += [conf_from_consistency_error]
        return ret

    def switch_to_train(self):
        self.flow_net.train()

    def switch_to_eval(self):
        self.flow_net.eval()


def get_combi_list(num_views, method='all') -> torch.tensor:
    """Compute list of image pairs. 
    Args:
        num_views int): number of total views
        method (str, optional): _description_. Defaults to 'random'.

    Returns:
        torch.tensor: list of image pair indexes, in format (2, N)
    """
    if method == 'all':
        combi_list = permutations(range(num_views), 2)
        # if num_views=10, 
        # [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), 
        # (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), 
        # (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),...]
        # choose num_neighbors ones
        combi_list = np.array(list(combi_list)).astype(np.int_).T
        # 2x(num_views*(num_views - 1)). [[0, 0, 0, ... 1, 1, 1], [1, 2, 3, ..9, 0, 2, 3, 4, 5, ]]
        # all combinations except for oneself
        assert combi_list.shape[-1] == num_views * (num_views - 1)
    elif method == 'random':
        # choose for each 1 view
        combi_list = np.stack((np.arange(num_views), np.random.permutation(num_views))).astype(np.int_)  # 2x10
    else:
        raise

    return torch.from_numpy(combi_list)


def flow_net_model_select(backbone, train_features=False):
    if backbone == 'PDCNet':
        global_optim_iter=3
        local_optim_iter=3
        from third_party.DenseMatching.models.PDCNet.PDCNet import PDCNet_vgg16

        global_gocor_arguments = {'optim_iter': global_optim_iter, 'steplength_reg': 0.1,
                                    'train_label_map': False, 'apply_query_loss': True,
                                    'reg_kernel_size': 3, 'reg_inter_dim': 16,
                                    'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}

        flow_net = PDCNet_vgg16(
            global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
            normalize='leakyrelu', same_local_corr_at_all_levels=True,
            local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
            local_decoder_type='OpticalFlowEstimatorResidualConnection',
            global_decoder_type='CMDTopResidualConnection',
            corr_for_corr_uncertainty_decoder='corr', train_features=train_features, 
            give_layer_before_flow_to_uncertainty_decoder=True,
            var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0)

    elif backbone == 'SPSG':
        from source.utils.spsg_matcher.superglue_module import SPSGInference
        flow_net = SPSGInference()
        compute_flow = True
    elif backbone == 'RoMa_indoor':
        weights = os.path.join(str(Path(__file__).parent / '../../checkpoint/roma_indoor.pth'))
        weights = torch.load(weights)
        from third_party.roma.roma import roma_indoor
        flow_net = roma_indoor(device=torch.device("cuda"), weights=weights)
        flow_net.symmetric = False
    elif backbone == 'RoMa_outdoor':
        weights = os.path.join(str(Path(__file__).parent / '../../checkpoint/roma_outdoor.pth'))
        weights = torch.load(weights)
        from third_party.roma.roma import roma_outdoor
        flow_net = roma_outdoor(device=torch.device("cuda"), weights=weights)
        flow_net.symmetric = False
    else:
        raise NotImplemented()
    return flow_net


def visualize_mapping_combinations(images, mapping_est, batched_conf_map, combi_list, save_path=None, min_conf=0.8):
    # flow_est [N, 2, H, W] where N is combi_list.shape[1]
    mapping_est = mapping_est.detach()
    batched_conf_map = batched_conf_map.detach()
    H, W = mapping_est.shape[-2:]
    mapping_est = mapping_est.cpu().numpy()
    # images [n_views, 3, H, W]
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    batched_conf_map = batched_conf_map.squeeze(1).cpu().numpy()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import math

    n_flows = combi_list.shape[1]
    def plot_mapping_set(n_start, n_end):
        w = 10
        n_flows = n_end - n_start
        h = w * math.ceil(n_flows / 4.)
        fig = plt.figure(figsize=(w, h), tight_layout=True)
        spec2 = gridspec.GridSpec(ncols=4, nrows=n_flows, figure=fig)
        for ind, i in enumerate(range(n_start, n_end)):
            i_self, i_other_img = combi_list[:, i]
            row_nu = ind
            image_target = images[i_self]
            image_source = images[i_other_img]
            warped = remap_using_correspondence_map(image_source, mapping_est[i, 0], mapping_est[i, 1])

            plt.subplot(spec2[row_nu, 0])
            plt.imshow(image_source)
            plt.title(f'Source Image {i_other_img}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 1])
            plt.imshow(image_target)
            plt.title(f'Target Image, {i_self}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 2])
            plt.imshow(warped)
            plt.title(f'Warped source {i_other_img} to  {i_self}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 3])
            plt.imshow(batched_conf_map[i])
            plt.title(f'conf map {i_other_img} to  {i_self}, {(batched_conf_map[i] > min_conf).sum()} conf px')
            plt.axis("off")

        fig.tight_layout(pad=0)
        canvas = FigureCanvas(fig)
        canvas.draw()      
        # draw the canvas, cache the renderer
        width, height = canvas.get_width_height() #fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close()
        return image 
    
    all_images = []
    for start_i in range(0, n_flows, 50):
        all_images.append(plot_mapping_set(start_i, min(n_flows, start_i+50)))

    def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    def stack_images_rows_with_pad(list_of_images):
        max_h = list_of_images[0].shape[0]
        return [pad_along_axis(x, max_h, axis=0) for x in list_of_images]

    if len(all_images) > 1:
        all_images = stack_images_rows_with_pad(all_images)
        image = np.concatenate(all_images, axis=1)
    else:
        image = all_images[0]
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Written image to {}'.format(save_path))
    return image
