import numpy as np
import torch
import cv2
from typing import Union, Any, List, Callable, Dict

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def resize(image: np.ndarray, size: List[int], fn: Callable[[List], float]=None, 
           interp: str='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        # pil
        w, h = image.size[:2]
        
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
    else:
        raise ValueError(f'Incorrect new size: {size}')
    
    # wants size to be dividable by 2!
    # h_new = h_new + 1 if h_new % 2 == 1 else h_new
    # w_new = w_new + 1 if w_new % 2 == 1 else w_new
    scale = (w_new / w, h_new / h)
    
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)
    else:
        image = image.resize((w_new, h_new))
    return image, scale

def numpy_image_to_torch(image: np.ndarray):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy((image / 255.).astype(np.float32, copy=False))
