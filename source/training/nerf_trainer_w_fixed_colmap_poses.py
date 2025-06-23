
import torch
import torch.nn as nn
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.training.engine.iter_based_trainer import get_log_string
import source.training.nerf_trainer as nerf
from source.models.renderer import Graph
import source.utils.camera as camera
from source.training.joint_pose_nerf_trainer import CommonPoseEvaluation

class InitialPoses(torch.nn.Module):
    def __init__(self, opt: Dict[str, Any], nbr_poses: int, initial_poses_w2c: torch.Tensor,  
                 device: torch.device):
        super().__init__()

        self.opt = opt
        self.nbr_poses = nbr_poses  # corresponds to the total number of poses!
        self.device = device
        self.initial_poses_w2c = initial_poses_w2c  # corresponds to initialization of all the poses
        # including the ones that are fixed
        self.initial_poses_c2w = camera.pose.invert(self.initial_poses_w2c)

    def get_initial_w2c(self) -> torch.Tensor:
        return self.initial_poses_w2c
        
    def get_c2w_poses(self) -> torch.Tensor:
        return self.initial_poses_c2w

    def get_w2c_poses(self) -> torch.Tensor:
        return self.initial_poses_w2c


# ============================ computation graph for forward/backprop ============================

class Graph(Graph):
    """NeRF (mlp + rendering) system when considering fixed noisy poses. """
    def __init__(self, opt: Dict[str, Any], device: torch.device, 
                 pose_net: Any):
        super().__init__(opt, device)

        # nerf networks already done 
        self.pose_net = pose_net

    def get_w2c_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode=None) -> torch.Tensor:
        if mode=="train":
            pose = self.pose_net.get_w2c_poses()
        elif mode in ["val","eval","test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end 
            # align test pose to refined coordinate system (up to sim3)

            # the poses were aligned at the beginning and fixed, they do not need further alignement
            pose = data_dict.pose

            # Because the fixed poses might be noisy, here can learn an extra alignement on top
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([data_dict.pose_refine_test,pose])
        else: 
            raise ValueError
        return pose

    def get_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        return self.get_w2c_pose(opt, data_dict, mode)

    def get_c2w_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        w2c = self.get_w2c_pose(opt, data_dict, mode)
        return camera.pose.invert(w2c)