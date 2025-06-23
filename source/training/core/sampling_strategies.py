import torch
from typing import List, Tuple, Union, Dict, Any, Optional


class RaySamplingStrategy:
    """Computes which rays we should sample.

    There are multiple ways of sampling: 
        - either randomly everywhere (except in border of size patch size)  
        - the ones that hit the dilated foreground mask.
        - Within a center bounding box

    Args:
        opt: settings
        data_dict: the data dict.
        device
    """
    def __init__(self, opt: Dict[str, Any], data_dict: Dict[str, Any], device: torch.device):
        self.opt = opt
        self.device = device


        self.nbr_images, _, self.H, self.W = data_dict.image.shape

        self.all_possible_pixels = self.get_all_samples(data_dict)  # [HW, 2]
        # rays = self.all_possible_pixels[:, 1] * self.W + self.all_possible_pixels[:, 0]

        self.all_center_pixels = self.get_all_center_pixels(data_dict)  # [N, 2]

        assert self.opt.sample_fraction_in_fg_mask == 0

        # pixels in patch
        y_range = torch.arange(self.opt.depth_regu_patch_size, dtype=torch.long,device=self.device)
        x_range = torch.arange(self.opt.depth_regu_patch_size, dtype=torch.long,device=self.device)
        Y,X = torch.meshgrid(y_range,x_range) # [patch_size,patch_size]
        self.dxdy = torch.stack([X,Y],dim=-1).view(-1,2) # [patch_size**2,2]

    
    def get_all_samples(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Samples all pixels/rays """
        H, W = data_dict.image.shape[-2:]
        if self.opt.loss_weight.depth_patch is not None:
            # exclude the patch size 
            y_range = torch.arange(H-self.opt.depth_regu_patch_size-1,dtype=torch.long,device=self.device)
            x_range = torch.arange(W-self.opt.depth_regu_patch_size-1,dtype=torch.long,device=self.device)
        else:
            y_range = torch.arange(H,dtype=torch.long,device=self.device)
            x_range = torch.arange(W,dtype=torch.long,device=self.device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
        # ray_idx = Y*W + X  # [HW]
        return xy_grid.long()  # ray_idx.view(-1).long()

    def get_all_center_pixels(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Sample all pixels/rays within center bounding box"""
        H, W = data_dict.image.shape[-2:]
        dH = int(H//2 * self.opt.precrop_frac)
        dW = int(W//2 * self.opt.precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        coords = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        return coords.long().to(self.device)


def sample_rays(H: int, W: int, precrop_frac: float=0.5,
                fraction_in_center: float=0., nbr: int=None, seed=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample pixels/rays within the image, the output formatting is (N, 2)/(N)"""

    if seed is not None:
        torch.manual_seed(seed)

    # all pixels
    y_range = torch.arange(H - 1,dtype=torch.long)
    x_range = torch.arange(W - 1,dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2).long() # [HW,2]
    n = xy_grid.shape[0]
    x_ind = xy_grid[..., 0]
    y_ind = xy_grid[..., 1]

    if fraction_in_center > 0.:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        pixels_in_center = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        if nbr is not None:
            nbr_center = int(nbr*fraction_in_center)
            nbr_all = nbr - nbr_center
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr_all]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]

            idx = torch.randperm(len(pixels_in_center), device=x_ind.device)[:nbr_center]
            pixels_in_center = pixels_in_center[idx] # [N, 2]
            x_ind = torch.cat((x_ind, pixels_in_center[..., 0]))
            y_ind = torch.cat((y_ind, pixels_in_center[..., 1]))  # 
            n = len(x_ind)
    else:
        if nbr is not None:
            # select a subset of those
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]
            
    n = len(x_ind)
    pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(n, -1)

    rays = pixel_coords[..., 1] * W + pixel_coords[..., 0]

    return pixel_coords.float(), rays  # (N, 2), (N)