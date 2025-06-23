import torch
from typing import Any, Tuple, Dict
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img_and_check_depth
from source.utils.geometry.geometric_utils_numpy import get_absolute_coordinates


class CorrrespondenceUtils:
    """Basic correspondence operations. """

    def get_gt_correspondence_maps_all_to_all(self, n_views: int) -> torch.Tensor:
        """For debugging, creates the ground-truth correspondence maps relating the images, using the
        ground-truth depths and ground-truth poses. 
        outpit is (n_views, n_views, 3, H, W). First 2 channels are the correspondence map, last channel is the 
        valid mask. Exhaustive matching of all views to all views (including itself). """
        all_corres_map_and_mask = []
        for id_self in range(n_views):
            corres_map_and_mask_from_id_self = []
            for id_matching_view in range(n_views):
                if id_self == id_matching_view:
                    corres_map = self.grid.unsqueeze(0)  # (1, H, W, 2)
                    mask = torch.ones_like(corres_map[:, :, :, :1])  # (1, H, W, 1)
                    corres_map_and_mask_from_id_self.append(torch.cat((corres_map, mask), dim=-1))
                else:
                    corres_map, mask = get_correspondences_gt(self.train_data.all, idx_target=id_self, idx_source=id_matching_view)
                    # corres_map is (H, W, 2) and mask is (H, W)
                    corres_map_and_mask_from_id_self.append(torch.cat((corres_map, mask.unsqueeze(-1).float()), dim=-1).unsqueeze(0))
            corres_map_and_mask_from_id_self = torch.cat(corres_map_and_mask_from_id_self, dim=0)  # (N, H, W, 3)
            all_corres_map_and_mask.append(corres_map_and_mask_from_id_self.unsqueeze(0))
        corres_map_and_mask = torch.cat(all_corres_map_and_mask, dim=0).permute(0, 1, 4, 2, 3)  # (N, N, H, W, 3) and then (N, N, 3, H, W)
        return corres_map_and_mask


# ---------------------- Flow/correspondence processing utils -----------------------------

def get_mask_valid_from_conf_map(p_r: torch.Tensor, corres_map: torch.Tensor, 
                                 min_confidence: float, max_confidence: float=None) -> torch.Tensor:
    channel_first = False
    if len(corres_map.shape) == 4:
        # (B, 2, H, W) or (B, H, W, 2)
        if corres_map.shape[1] == 2:
            corres_map = corres_map.permute(0, 2, 3, 1)
            channel_first = True
        if len(p_r.shape) == 3:
            p_r = p_r.unsqueeze(-1)
        if p_r.shape[1] == 1:
            p_r = p_r.permute(0, 2, 3, 1)
        h, w = corres_map.shape[1:3]
        valid_matches = corres_map[:, :, :, 0].ge(0) & corres_map[:, :, :, 0].le(w-1) & corres_map[:, :, :, 1].ge(0) & corres_map[:, :, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (B, H, W, 1)
        if channel_first:
            mask = mask.permute(0, 3, 1, 2)
    else:
        if corres_map.shape[0] == 2:
            corres_map = corres_map.permute(1, 2, 0)
        if len(p_r.shape) == 2:
            p_r = p_r.unsqueeze(-1)
            channel_first = True
        if p_r.shape[0] == 1:
            p_r = p_r.unsqueeze(1, 2, 0)
        h, w = corres_map.shape[:2]
        valid_matches = corres_map[:, :, 0].ge(0) & corres_map[:, :, 0].le(w-1) & corres_map[:, :, 1].ge(0) & corres_map[:, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (H, W, 1)
        if channel_first:
            mask = mask.permute(2, 0, 1)
    return mask 

# ------------------------------- debugging with ground-truth ---------------------

def get_correspondences_gt(data_dict: Dict[str, Any], idx_target: int, 
                           idx_source: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Computes ground-truth correspondence map using gt depth  map and poses. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
    """
    H, W = data_dict['image'].shape[-2:]
    pixels_target = get_absolute_coordinates(H, W, True).reshape(-1, 2).cuda()  # (H, W, 2)
    depth_source = data_dict['depth_gt'][idx_source].reshape(H, W)
    
    depth_target = data_dict['depth_gt'][idx_target].reshape(-1)
    valid_depth_target = data_dict['valid_depth_gt'][idx_target].reshape(H, W)

    K_target = data_dict['intr'][idx_target]
    K_source = data_dict['intr'][idx_source]
    w2c_target_ = data_dict['pose'][idx_target]
    w2c_target = torch.eye(4).cuda()
    w2c_target[:3] = w2c_target_

    w2c_source_ = data_dict['pose'][idx_source]
    w2c_source = torch.eye(4).cuda()
    w2c_source[:3] = w2c_source_

    target2source = w2c_source @ pose_inverse_4x4(w2c_target)
    repr_in_source, visible = batch_project_to_other_img_and_check_depth(
        pixels_target, di=depth_target, depthj=depth_source, Ki=K_target, Kj=K_source, 
        T_itoj=target2source, validi=valid_depth_target.reshape(-1), rth=0.05)
    corres_target_to_source = repr_in_source.reshape(H, W, 2)
    valid = corres_target_to_source[:, :, 0].ge(0) & corres_target_to_source[:, :, 1].ge(0) & \
        corres_target_to_source[:, :, 0].le(W-1) & corres_target_to_source[:, :, 1].le(H-1)
    valid = valid & valid_depth_target & visible.reshape(H, W)

    '''
    target = data_dict['image'][idx_target].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    source = data_dict['image'][idx_source].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    corres_map = corres_target_to_source.cpu().numpy()
    warped = remap_using_correspondence_map(source, corres_map.astype(np.float32)[:, :, 0], corres_map.astype(np.float32)[:, :, 1])
    image = np.concatenate((source, target, warped), axis=1)
    import cv2
    cv2.imwrite('test_img_{}_to_{}.png'.format(idx_target, idx_source), image * 255)
    '''
    return corres_target_to_source, valid




