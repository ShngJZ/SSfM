import copy
import torch
from typing import Any, Dict
from source.training.core.base_losses import Loss
from source.training.core.corres_loss import CorrespondencesPairRenderDepthAndGet3DPtsAndReproject
from source.training.core.triangulation_loss import TriangulationLoss

def define_loss(
        loss_type: str,
        opt: Dict[str, Any],
        nerf_net: torch.nn.Module,
        train_data: Dict[str, Any],
        device: torch.device,
        flow_net: torch.nn.Module=None
):
    loss_module = []
    if 'triangulation' in loss_type:
        corres_initializer = None
        for x in loss_module:
            if isinstance(x, CorrespondencesPairRenderDepthAndGet3DPtsAndReproject):
                corres_initializer = x
                break

        if corres_initializer is None:
            corres_initializer = CorrespondencesPairRenderDepthAndGet3DPtsAndReproject\
            (opt, nerf_net, flow_net=flow_net, train_data=train_data, device=device)

        corres_estimate_bundle = [
            copy.deepcopy(corres_initializer.corres_maps),
            copy.deepcopy(corres_initializer.flow_pairs),
            copy.deepcopy(corres_initializer.conf_maps),
            copy.deepcopy(corres_initializer.mask_valid_corr)
        ]
        loss_module.append(TriangulationLoss(opt, nerf_net, corres_estimate_bundle=corres_estimate_bundle, train_data=train_data, device=device))


        if 'depth_est' in train_data.all:
            nerf_net.pose_net.register_inputs(
                corres_estimate_bundle, train_data.all, norm_threshold_adjuster=opt.norm_threshold_adjuster, visibility_threshold=opt.loss_triangulation.visibility_threshold
            )


    loss_module = Loss(loss_module)
    return loss_module