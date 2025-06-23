import numpy as np
import torch
from typing import Union

def get_absolute_coordinates(h_scale: int, w_scale: int, 
                             is_torch_tensor: bool =False) -> Union[np.ndarray, torch.Tensor]:
    """Get pixels coordinates

    Args:
        h_scale (int)
        w_scale (int)
        is_torch_tensor (bool, optional): Defaults to False.

    Returns:
        grid (torch.Tensor): Pixels coordinates, (H, W, 2)
    """
    if is_torch_tensor:
        xx = torch.arange(start=0, end=w_scale).view(1, -1).repeat(h_scale, 1)
        yy = torch.arange(start=0, end=h_scale).view(-1, 1).repeat(1, w_scale)
        xx = xx.view(h_scale, w_scale, 1)
        yy = yy.view(h_scale, w_scale, 1)
        grid = torch.cat((xx, yy), -1).float()  # H, W, 2
    else:
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        grid = np.dstack((X, Y))
    return grid