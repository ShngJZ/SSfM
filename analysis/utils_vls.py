import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional
from torch import Tensor

def tensor2disp(tensor: Tensor, 
                vmax: float = 0.18, 
                percentile: Optional[float] = None, 
                viewind: int = 0) -> Image.Image:
    """
    Convert a depth tensor to a visualization image using the 'magma' colormap.

    Args:
        tensor: Input depth tensor
        vmax: Maximum value for normalization
        percentile: If provided, use this percentile of non-zero values for normalization
        viewind: Index of the view to visualize

    Returns:
        PIL.Image: Colorized depth visualization
    """
    colormap = plt.get_cmap('magma')
    depth_map = tensor[viewind, 0, :, :].detach().cpu().numpy()
    
    if percentile is not None:
        non_zero_values = depth_map > 0
        if np.sum(non_zero_values) > 100:
            vmax = np.percentile(depth_map[non_zero_values], 95)
        else:
            vmax = 1.0
    
    normalized_depth = depth_map / vmax
    colored_depth = (colormap(normalized_depth) * 255).astype(np.uint8)
    
    return Image.fromarray(colored_depth[:, :, 0:3])

def tensor2rgb(tensor: Tensor, viewind: int = 0) -> Image.Image:
    """
    Convert a tensor to an RGB image.

    Args:
        tensor: Input tensor in (B, C, H, W) format
        viewind: Index of the view to visualize

    Returns:
        PIL.Image: RGB image
    """
    rgb_array = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()
    rgb_array = rgb_array[viewind].numpy()
    
    if np.max(rgb_array) <= 2:
        rgb_array = rgb_array * 255
    
    rgb_array = np.clip(rgb_array, a_min=0, a_max=255).astype(np.uint8)
    return Image.fromarray(rgb_array)
