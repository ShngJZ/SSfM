import torch
from easydict import EasyDict as edict
from typing import Any, Dict
from source.utils.config_utils import override_options
from source.training.core.base_corres_loss import CorrespondenceBasedLoss


class CorrespondencesPairRenderDepthAndGet3DPtsAndReproject(CorrespondenceBasedLoss):
    """The main class for the correspondence loss of SPARF. It computes the re-projection error
    between previously extracted correspondences relating the input views. The projection
    is computed with the rendered depth from the NeRF and the current camera pose estimates.
    """

    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module,
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(opt, nerf_net, flow_net, train_data, device)
        default_cfg = edict({'diff_loss_type': 'huber',
                             'compute_photo_on_matches': False,
                             'renderrepro_do_pixel_reprojection_check': False,
                             'renderrepro_do_depth_reprojection_check': False,
                             'renderrepro_pixel_reprojection_thresh': 10.,
                             'renderrepro_depth_reprojection_thresh': 0.1,
                             'use_gt_depth': False,  # debugging
                             'use_gt_correspondences': False,  # debugging
                             'use_dummy_all_one_confidence': False  # debugging
                             })
        self.opt = override_options(self.opt, default_cfg)
        self.opt = override_options(self.opt, opt)
        self.loss_tag = 'corres'