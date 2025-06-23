import numpy as np
import torch
from typing import List, Union, Optional
from numpy.typing import NDArray

def scale_invariant(ground_truth: Union[NDArray, torch.Tensor], 
                   prediction: Union[NDArray, torch.Tensor],
                   min_depth: float = 0.1) -> float:
    """
    Compute the scale invariant loss based on differences of logs of depth maps.

    Args:
        ground_truth: Ground truth depth map
        prediction: Predicted depth map
        min_depth: Minimum depth threshold for valid pixels

    Returns:
        float: Scale invariant distance between the depth maps
    """
    ground_truth = ground_truth.reshape(-1)
    prediction = prediction.reshape(-1)

    valid_mask = ground_truth > min_depth
    ground_truth = ground_truth[valid_mask]
    prediction = prediction[valid_mask]

    log_diff = np.log(ground_truth) - np.log(prediction)
    num_pixels = np.float32(log_diff.size)

    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - 
                  np.square(np.sum(log_diff)) / np.square(num_pixels))

def compute_depth_errors(ground_truth: Union[NDArray, torch.Tensor], 
                       prediction: Union[NDArray, torch.Tensor],
                       min_depth: float = 0.0) -> List[float]:
    """
    Compute various depth estimation error metrics.

    Args:
        ground_truth: Ground truth depth map
        prediction: Predicted depth map
        min_depth: Minimum depth threshold for valid pixels

    Returns:
        List[float]: List of error metrics in the following order:
            [scale_invariant, log10, silog, abs_rel, sq_rel, 
             rms, log_rms, d05, d1, d2, d3]
            - scale_invariant: Scale invariant error
            - log10: Mean log10 error
            - silog: Scale invariant logarithmic error
            - abs_rel: Absolute relative error
            - sq_rel: Square relative error
            - rms: Root mean square error
            - log_rms: Log root mean square error
            - d05: Delta accuracy with threshold 1.25^0.5
            - d1: Delta accuracy with threshold 1.25
            - d2: Delta accuracy with threshold 1.25^2
            - d3: Delta accuracy with threshold 1.25^3
    """
    if len(ground_truth) == 0 or len(prediction) == 0:
        return [np.nan] * 11

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    valid_mask = ground_truth > min_depth
    ground_truth = ground_truth[valid_mask]
    prediction = prediction[valid_mask]

    if len(ground_truth) == 0 or len(prediction) == 0:
        return [np.nan] * 11

    # Compute threshold accuracy
    thresh = np.maximum((ground_truth / prediction), (prediction / ground_truth))
    DELTA_THRESHOLD = 1.25
    
    d05 = (thresh < DELTA_THRESHOLD ** 0.5).mean()
    d1 = (thresh < DELTA_THRESHOLD).mean()
    d2 = (thresh < DELTA_THRESHOLD ** 2).mean()
    d3 = (thresh < DELTA_THRESHOLD ** 3).mean()

    # Compute RMS errors
    rms = np.sqrt(np.mean((ground_truth - prediction) ** 2))
    log_rms = np.sqrt(np.mean((np.log(ground_truth) - np.log(prediction)) ** 2))

    # Compute relative errors
    abs_rel = np.mean(np.abs(ground_truth - prediction) / ground_truth)
    sq_rel = np.mean(((ground_truth - prediction) ** 2) / ground_truth)

    # Compute log errors
    log_diff = np.log(prediction) - np.log(ground_truth)
    silog = np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2) * 100

    log10_diff = np.abs(np.log10(prediction) - np.log10(ground_truth))
    log10 = np.mean(log10_diff)

    # Compute scale invariant error
    sc_inv = scale_invariant(ground_truth, prediction)

    return [sc_inv, log10, silog, abs_rel, sq_rel, rms, log_rms, d05, d1, d2, d3]
