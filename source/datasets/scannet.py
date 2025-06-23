import os
import io
import copy
import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import natsort
from PIL import Image

from source.datasets.base import Dataset
from source.datasets.data_utils import numpy_image_to_torch, resize
from source.utils.helper import h5_open_wait

class ScanNet(Dataset):
    """ScanNet dataset for NeRF training and evaluation.
    
    This dataset handles loading and processing of ScanNet scenes, including RGB images,
    camera poses, intrinsics, and depth maps.
    """
    
    # Default image dimensions
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640
    
    # Depth range for the dataset
    DEPTH_RANGE = torch.tensor([0.5, 10.0], dtype=torch.float32)
    
    # Frame window size for sampling
    FRAME_WINDOW = 25
    
    def __init__(self, args: Dict[str, Any], split: str, scenes: str = '', **kwargs):
        """Initialize the ScanNet dataset.
        
        Args:
            args: Configuration dictionary containing dataset parameters
            split: Dataset split ('train' or 'test')
            scenes: Scene identifier
            **kwargs: Additional arguments
        """
        super().__init__(args, split)

        self.base_dir = args.env.scannet
        self.scene = scenes
        
        assert args.train_sub >= 3, "train_sub must be at least 3"
        assert np.mod(args.train_sub, 2) == 1, "train_sub must be odd"

        # Generate deterministic seed from scene name
        seed = int(hashlib.sha1(scenes.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        h5pypath = os.path.join(self.base_dir, scenes, '{}.hdf5'.format(scenes))
        hf = h5_open_wait(h5pypath)
        assert hf is not None
        if hf is not None:
            # Process dataset entry to get image list
            entry = args.dataset_entry.replace("'", "").rstrip('\n')
            images = entry.split(' ')[1:]  # Skip first component
            images_ordered = natsort.natsorted(images)

            # Read Poses and Intrinsics
            poses, intrinsics = self.read_poses_intrinsics(images, hf)
            self.render_rgb_files = images
            self.render_poses_c2w = poses
            self.render_intrinsics = intrinsics
            self.render_img_id = [int(x.split('.')[0]) for x in images]

        self.ht, self.wd = self.IMAGE_HEIGHT, self.IMAGE_WIDTH
        print(f"Dataset contains {len(self.render_rgb_files)} images")
        print(f"Train Images: {' '.join(images_ordered)}")
        
        self.depth_range = self.DEPTH_RANGE

    def read_poses_intrinsics(self, images: List[str], h5pyfile) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Read camera poses and intrinsics from h5py file.
        
        Args:
            images: List of image filenames
            h5pyfile: Open h5py file containing pose and intrinsic data
            
        Returns:
            Tuple containing lists of camera-to-world poses and intrinsic matrices
        """
        poses_c2w = []
        intrinsics = []

        for image in images:
            stem_name = image.split('.')[0]
            pose_data = np.array(h5pyfile['pose'][f'{stem_name}.txt'])
            intrinsic_data = np.array(h5pyfile['intrinsic']['intrinsic_color.txt'])
            
            pose_c2w = self.read_scannet_pose(
                io.StringIO(io.BytesIO(pose_data).read().decode('UTF-8'))
            )
            intrinsic = self.read_scannet_intrinsic(
                io.StringIO(io.BytesIO(intrinsic_data).read().decode('UTF-8'))
            )
            
            poses_c2w.append(pose_c2w)
            intrinsics.append(intrinsic)

        return poses_c2w, intrinsics

    def read_scannet_pose(self, path) -> np.ndarray:
        """Read ScanNet's camera-to-world pose matrix.
        
        Args:
            path: File-like object containing pose data
            
        Returns:
            4x4 camera-to-world transformation matrix
        """
        return np.loadtxt(path, delimiter=' ')

    def read_scannet_intrinsic(self, path) -> np.ndarray:
        """Read ScanNet's camera intrinsic matrix.
        
        Args:
            path: File-like object containing intrinsic data
            
        Returns:
            3x3 camera intrinsic matrix
        """
        return np.loadtxt(path, delimiter=' ')[:-1, :-1]

    def load_depth(self, depth_ref) -> np.ndarray:
        """Load and process depth map.
        
        Args:
            depth_ref: File-like object containing depth image
            
        Returns:
            Processed depth map as numpy array
        """
        depth = np.array(Image.open(depth_ref)).astype(np.float32) / 1000
        depth, _ = resize(depth, (self.ht, self.wd))
        
        # Create and resize validity mask
        depth_valid = (depth > 0).astype(np.float32)
        depth_valid, _ = resize(depth_valid, (self.ht, self.wd))
        
        # Zero out unreliable depth values
        depth[depth_valid < 0.9] = 0
        return depth

    def load_rgb_intrinsic(self, rgb, K) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and process RGB image and adjust intrinsic matrix.
        
        Args:
            rgb: File-like object containing RGB image
            K: Original intrinsic matrix
            
        Returns:
            Tuple of processed RGB image tensor and scaled intrinsic matrix
        """
        image = Image.open(rgb)
        worg, horg = image.size
        
        # Resize and process image
        rgb, _ = resize(image, (self.ht, self.wd))
        rgb = np.array(rgb).astype(np.float32)
        rgb = self._add_white_border(rgb)
        rgb = numpy_image_to_torch(rgb)
        
        # Scale intrinsic matrix to match new image dimensions
        K = self._scale_intrinsic(K, self.wd, self.ht, worg, horg)
        return rgb, K

    def __len__(self):
        return len(self.render_rgb_files)

    def _add_white_border(self, rgb: np.ndarray, padding: int = 5) -> np.ndarray:
        """Add white border to RGB image.
        
        Args:
            rgb: Input RGB image
            padding: Border width in pixels
            
        Returns:
            RGB image with white border
        """
        h, w, _ = rgb.shape
        mask = np.zeros_like(rgb)
        mask[padding:h-padding, padding:w-padding, :] = 1.0
        rgb = rgb * mask + 255.0 * (1 - mask)
        return np.clip(rgb, 0.0, 255.0)

    def _scale_intrinsic(self, K: np.ndarray, wtarget: int, htarget: int, 
                        worg: int, horg: int) -> np.ndarray:
        """Scale intrinsic matrix to match new image dimensions.
        
        Args:
            K: Original intrinsic matrix
            wtarget: Target width
            htarget: Target height
            worg: Original width
            horg: Original height
            
        Returns:
            Scaled intrinsic matrix
        """
        sx, sy = wtarget / worg, htarget / horg
        scale_matrix = np.eye(3)
        scale_matrix[0, 0], scale_matrix[1, 1] = sx, sy
        return scale_matrix @ K

    def get_all_camera_poses(self, args) -> torch.Tensor:
        """Get all camera poses in world-to-camera format.
        
        Args:
            args: Configuration arguments
            
        Returns:
            Tensor containing all camera poses
        """
        poses_c2w = np.stack(copy.deepcopy(self.render_poses_c2w), axis=0)
        return torch.inverse(torch.from_numpy(poses_c2w).float())[:, 0:3, :]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item by index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - idx: Image index
                - rgb_path: Path to RGB image
                - image: RGB image tensor [3, H, W], normalized to [0, 1]
                - intr: Camera intrinsic matrix [3, 3]
                - pose: World-to-camera transformation matrix [3, 4]
                - depth_range: Valid depth range [1, 2]
                - scene: Scene identifier
                - depth_gt: Ground truth depth map [H, W]
                - valid_depth_gt: Depth validity mask [H, W]
        """
        rgb_file = self.render_rgb_files[idx]
        render_pose_c2w = self.render_poses_c2w[idx]
        render_pose_w2c = np.linalg.inv(render_pose_c2w)
        render_intrinsics = self.render_intrinsics[idx]
        scene = self.scene

        h5pypath = os.path.join(self.base_dir, scene, '{}.hdf5'.format(scene))
        hf = h5_open_wait(h5pypath)
        assert hf is not None
        if hf is not None:
            stem_name = rgb_file.split('.')[0]
            rgb, render_intrinsics = self.load_rgb_intrinsic(
                io.BytesIO(np.array(hf['color'][f'{stem_name}.jpg'])), K=render_intrinsics
            )
            depth_gt = self.load_depth(
                io.BytesIO(np.array(hf['depth'][f'{stem_name}.png']))
            )

        valid_depth_gt = depth_gt > 0.

        assert depth_gt.shape[:2] == rgb.shape[1::]
        assert valid_depth_gt.shape[:2] == rgb.shape[1::]
        ret = {
            'idx': idx,
            "rgb_path": rgb_file,
            'depth_gt': depth_gt,  # (H, W)
            'valid_depth_gt': valid_depth_gt,  # (H, W)
            'image': rgb,  # torch tensor 3, self.H, self.W
            'intr': render_intrinsics[:3, :3].astype(np.float32),
            'pose': render_pose_w2c[:3].astype(np.float32),  # 3x4, world to camera
            "depth_range": self.depth_range,
            'scene': self.scene
        }
        return ret
