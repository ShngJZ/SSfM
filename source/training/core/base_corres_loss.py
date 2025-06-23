import numpy as np
import torch
from easydict import EasyDict as edict
from typing import Any, Dict
from source.training.core.base_losses import BaseLoss
from source.training.core.correspondence_utils import (CorrrespondenceUtils, get_mask_valid_from_conf_map)
from source.utils.config_utils import override_options

class CorrespondenceBasedLoss(BaseLoss, CorrrespondenceUtils):
    """Correspondence Loss. Main signal for the joint pose-NeRF training. """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(device=device)
        default_cfg = edict({'matching_pair_generation': 'all', 
                             'min_nbr_matches': 500, 
                            
                             'pairing_angle_threshold': 30, # degree, in case 'angle' pair selection chosen
                             'filter_corr_w_cc': False, 
                             'min_conf_valid_corr': 0.95, 
                             'min_conf_cc_valid_corr': 1./(1. + 1.5), 
                             })
        self.opt = override_options(default_cfg, opt)
        self.device = device

        self.train_data = train_data
        H, W = train_data.all.image.shape[-2:]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        self.grid = torch.stack((xx, yy), dim=-1).to(self.device).float()  # ( H, W, 2)
        self.grid_flat = self.grid[:, :, 1] * W + self.grid[:, :, 0]  # (H, W), corresponds to index in flattedned array (in H*W)
        self.grid_flat = self.grid_flat.to(self.device).long()


        self.net = nerf_net
        self.flow_net = flow_net

        self.gt_corres_map_and_mask_all_to_all = None
        if 'depth_gt' in train_data.all.keys():
            self.gt_corres_map_and_mask_all_to_all = self.get_gt_correspondence_maps_all_to_all(n_views=len(train_data))
            # (N, N, 3, H, W)

        self.compute_correspondences(train_data)
        # flow will either be computed on the fly (if weights are finetuned or not)

    @torch.no_grad()
    def compute_correspondences(self, train_data: Dict[str, Any]):
        """Compute correspondences relating the input views. 

        Args:
            train_data (dataset): training dataset. The keys all is a dictionary, 
                                  containing the entire training data. 
                                  train_data.all has keys 'idx', 'image', 'intr', 'pose' 
                                  and all images of the scene already stacked here.

        """
        print('Computing flows')
        images = train_data.all['image']  # (N_views, 3, H, W)
        H, W = images.shape[-2:]
        poses = train_data.all['pose']  # ground-truth poses w2c
        n_views = images.shape[0]

        if self.opt.matching_pair_generation == 'all':
            # exhaustive pairs, but (1, 2) or (2, 1) are treated as the same. 
            combi_list = generate_pair_list(n_views)
        elif self.opt.matching_pair_generation == 'all_to_all':
            # all pairs, including both directions. (1, 2) and (2, 1)
            combi_list = self.flow_net.combi_list  # 2xN
            # first row is target, second row is source
        elif self.opt.matching_pair_generation == 'angle':
            # pairs such as the angular distance between images is below a certain threshold
            combi_list = image_pair_candidates_with_angular_distance\
                (poses, pairing_angle_threshold=self.opt.pairing_angle_threshold)
        else:
            raise ValueError

        print(f'Computing {combi_list.shape[1]} correspondence maps')
        if combi_list.shape[1] == 0:
            self.flow_plot, self.flow_plot_masked = None, None
            self.corres_maps, self.conf_maps, self.mask_valid_corr = None, None, None
            self.filtered_flow_pairs = []
            return 

        # IMPORTANT: the batch norm should be to eval!!
        # otherwise, the statistics of the image are too different and it does something bad!
        if self.opt.filter_corr_w_cc:
            corres_maps, conf_maps, conf_maps_from_cc, flow_plot = self.flow_net.compute_flow_and_confidence_map_and_cc_of_combi_list\
                (images, combi_list_tar_src=combi_list, plot=True,
                use_homography=self.opt.use_homography_flow) 
        else:
            corres_maps, conf_maps, flow_plot = self.flow_net.compute_flow_and_confidence_map_of_combi_list(
                images, combi_list_tar_src=combi_list, plot=True, use_homography=self.opt.use_homography_flow, additional_data=train_data.all
            )
        mask_valid_corr = get_mask_valid_from_conf_map(p_r=conf_maps.reshape(-1, 1, H, W), 
                                                       corres_map=corres_maps.reshape(-1, 2, H, W), 
                                                       min_confidence=self.opt.min_conf_valid_corr)  # (n_views*(n_views-1), 1, H, W)

        if self.opt.filter_corr_w_cc:
            mask_valid_corr = mask_valid_corr & conf_maps_from_cc.ge(self.opt.min_conf_cc_valid_corr)
        
        # save the flow examples for tensorboard
        self.flow_plot = flow_plot  
        self.flow_plot_masked = None
        flow_plot = self.flow_net.visualize_mapping_combinations(images=train_data.all.image, 
                                                                 mapping_est=corres_maps.reshape(-1, 2, H, W), 
                                                                 batched_conf_map=mask_valid_corr.float(), 
                                                                 combi_list=combi_list, save_path=None)
        flow_plot = torch.from_numpy( flow_plot.astype(np.float32)/255.).permute(2, 0, 1)
        self.flow_plot_masked = flow_plot

        # when we only computed a subset
        self.corres_maps = corres_maps  # (combi_list.shape[1], 3, H, W)
        self.conf_maps = conf_maps
        self.mask_valid_corr = mask_valid_corr
        # should be list of the matching index for each of the image. 
        # first row/element corresponds to the target image, second is the source image
        flow_pairs = (combi_list.cpu().numpy().T).tolist()  
        self.flow_pairs = flow_pairs
        # list of pairs, the target is the first element, source is second
        assert self.corres_maps.shape[0] == len(flow_pairs)

        # keep only the correspondences for which there are sufficient confident regions 
        filtered_flow_pairs = []
        for i in range(len(flow_pairs)):
            nbr_confident_regions = self.mask_valid_corr[i].sum()
            if nbr_confident_regions > self.opt.min_nbr_matches:
                filtered_flow_pairs.append((i, flow_pairs[i][0], flow_pairs[i][1]))
                # corresponds to index_of_flow, index_of_target_image, index_of_source_image
        self.filtered_flow_pairs = filtered_flow_pairs
        print(f'{len(self.filtered_flow_pairs)} possible flow pairs')
        return 
