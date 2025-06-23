from typing import Any, List, Union, Tuple
import torch
import numpy as np


def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points: Union[torch.Tensor, np.ndarray], eps=1e-6):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)




def sample_depth(pts: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """sample depth at points. 

    Args:
        pts (torch.Tensor): (N, 2)
        depth (torch.Tensor): (B, 1, H, W)
    """
    h, w = depth.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    batched = len(depth.shape) == 3
    if not batched:
        pts, depth = pts[None], depth[None]

    pts = (pts / pts.new_tensor([[w-1, h-1]]) * 2 - 1)[:, None]
    depth = torch.where(depth > 0, depth, depth.new_tensor(float('nan')))
    depth = depth[:, None]
    interp_lin = grid_sample(
            depth, pts, align_corners=True, mode='bilinear')[:, 0, 0]
    interp_nn = grid_sample(
            depth, pts, align_corners=True, mode='nearest')[:, 0, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = (~torch.isnan(interp)) & (interp > 0)
    # will exclude out of view matches here, except if the depth was dense and the points falls right at the border,
    # then nearest can get the depth value.
    if not batched:
        interp, valid = interp[0], valid[0]
    return interp, valid


def batch_project_to_other_img_and_check_depth(kpi: torch.Tensor, di: torch.Tensor, 
                                               depthj: torch.Tensor, 
                                               Ki: torch.Tensor, Kj: torch.Tensor, 
                                               T_itoj: torch.Tensor, 
                                               validi: torch.Tensor, 
                                               rth: float=0.1, 
                                               return_repro_error: bool=False
                                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project pixels of one image to the other, and run depth-check. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        depthj: depth map of image j, BxHxW
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        validi: BxN, Bool mask
        rth: percentage of acceptable depth reprojection error. 
        return_repro_error: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        visible: Bool mask, visible pixels that have a valid reprojection error, BxN
    """

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)
    repro_error = torch.abs(di_j - dj) / dj
    consistent = repro_error < rth
    visible = validi & consistent & validj
    if return_repro_error:
        return kpi_j, visible, repro_error
    return kpi_j, visible



def batch_backproject_to_3d(kpi: torch.Tensor, di: torch.Tensor, 
                            Ki: torch.Tensor, T_itoj: torch.Tensor) -> torch.Tensor:
    """
    Backprojects pixels to 3D space 
    Args:
        kpi: BxNx2 coordinates in pixels
        di: BxN, corresponding depths
        Ki: camera intrinsics, Bx3x3
        T_itoj: Bx4x4
    Returns:
        kpi_3d_j: 3D points in coordinate system j, BxNx3
    """

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
        to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    return kpi_3d_j  # Nx3


def batch_project(kpi_3d_i: torch.Tensor, T_itoj: torch.Tensor, Kj: torch.Tensor, return_depth=False):
    """
    Projects 3D points to image pixels coordinates. 
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        T_itoj: Bx4x4
        Kj: camera intrinsics Bx3x3

    Returns:
        pixels projections in image j, BxNx2
    """
    kpi_3d_in_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_2d_in_j = kpi_3d_in_j @ Kj.transpose(-1, -2)
    if return_depth:
        return from_homogeneous(kpi_2d_in_j), kpi_3d_in_j[..., -1]
    return from_homogeneous(kpi_2d_in_j)
