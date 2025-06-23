import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from source.utils import camera


def get_layer_dims(layers: torch.nn.Module) -> List[Tuple[int, int]]:
    """Get input/output dimensions for each layer.
    
    Args:
        layers: Sequential neural network layers
        
    Returns:
        List of (input_dim, output_dim) tuples for each layer
    """
    return list(zip(layers[:-1], layers[1:]))


def l2_normalize(x: torch.Tensor, eps: torch.Tensor = torch.finfo(torch.float32).eps) -> torch.Tensor:
    """Normalize tensor to unit length along last axis.
    
    Args:
        x: Input tensor
        eps: Small constant for numerical stability
        
    Returns:
        Unit normalized tensor
    """
    return x / torch.sqrt(
        torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
    )

    
class FrequencyEmbedder:
    """Positional encoding using frequency embeddings."""
    def __init__(self, opt: Dict[str, Any]):
        """Initialize frequency embedder.
        
        Args:
            opt: Configuration options dictionary
        """
        self.opt = opt

    def __call__(self, opt: Dict[str, Any], input: torch.Tensor, L: int) -> torch.Tensor:
        """Apply frequency embedding to input tensor.
        
        Embeds input using sinusoidal functions at different frequencies.
        Can use either logarithmic or linear frequency sampling.
        
        Args:
            opt: Configuration options
            input: Input tensor to embed [..., N]
            L: Number of frequency bands
            
        Returns:
            Frequency embedded tensor [..., 2NL]
        """
        shape = input.shape
        
        # Generate frequency bands
        if opt.arch.posenc.log_sampling:
            if opt.arch.posenc.include_pi_in_posenc:
                # Logarithmic sampling with pi scaling
                freq = 2**torch.arange(L, dtype=torch.float32, device=input.device) * np.pi
            else:
                # Standard logarithmic sampling
                freq = 2**torch.arange(L, dtype=torch.float32, device=input.device)
        else:
            # Linear sampling
            freq = torch.linspace(2.**0., 2.**(L-1), steps=L) * np.pi
            
        # Apply frequency encoding
        spectrum = input[..., None] * freq  # [..., N, L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [..., N, L]
        
        # Stack and reshape
        input_enc = torch.stack([sin, cos], dim=-2)  # [..., N, 2, L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [..., 2NL]
        
        return input_enc


class BaldNeRF(torch.nn.Module):
    """Neural Radiance Field (NeRF) implementation with feature volume."""
    
    def __init__(self, opt: Dict[str, Any], is_fine_network: bool = False):
        """Initialize BaldNeRF model.
        
        Args:
            opt: Configuration options dictionary
            is_fine_network: Whether this is a fine network in hierarchical sampling
        """
        super().__init__()
        self.opt = opt

        # Initialize feature volume
        cdim, dsh, dsw, depthdim = self.opt.nerf.feature_volume_size
        self.feature_volume = torch.nn.Parameter(
            torch.zeros([1, cdim, depthdim, 480 // dsh, 640 // dsw]).normal_(mean=0, std=0.01)
        )
        logging.info(
            f"Initialized Feature Volume: [1, {cdim}, {depthdim}, {480//dsh}, {640//dsw}]"
        )

        # Calculate input dimensions
        self.input_3D_dim = self._calculate_input_dims(opt)
        assert self.input_3D_dim > 0, "Input dimension must be positive"

        # Setup activation options
        self._setup_activation_options(opt)

    def _calculate_input_dims(self, opt: Dict[str, Any]) -> int:
        """Calculate input dimensions based on configuration.
        
        Args:
            opt: Configuration options
            
        Returns:
            Total input dimension
        """
        input_3D_dim = 0
        if opt.arch.posenc.add_raw_3D_points:
            input_3D_dim += 3
        if opt.arch.posenc.L_3D > 0:
            input_3D_dim += 6 * opt.arch.posenc.L_3D
        return input_3D_dim

    def _setup_activation_options(self, opt: Dict[str, Any]):
        """Setup activation function options.
        
        Args:
            opt: Configuration options
        """
        if 'unique_3D_act' in opt.nerf:
            self.unique_3D_act = opt.nerf.unique_3D_act
            self.unique_3D_act_temperature = opt.nerf.unique_3D_act_temperature
            cdim, dsh, dsw, _ = opt.nerf.feature_volume_size
            self.learnable_max_value = nn.Parameter(
                torch.ones([1, 1, 1, 480 // dsh, 640 // dsw]) * 5, 
                requires_grad=True
            )
        else:
            self.unique_3D_act = False

    def initialize(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.tensorflow_init_weights(self.opt, m)

    def choose_activation(self, opt: Dict[str, Any]) -> nn.Module:
        """Choose activation function.
        
        Args:
            opt: Configuration options
            
        Returns:
            Activation function module
        """
        return nn.ReLU(True)

    def tensorflow_init_weights(self, opt: Dict[str, Any], linear: nn.Module, out: Optional[str] = None):
        """Initialize weights using TensorFlow-style Xavier initialization.
        
        Args:
            opt: Configuration options
            linear: Linear or Conv2d layer to initialize
            out: Initialization mode ('all', 'first', or None)
        """
        relu_gain = nn.init.calculate_gain("relu")
        
        if out == "all":
            nn.init.xavier_uniform_(linear.weight)
        elif out == "first":
            nn.init.xavier_uniform_(linear.weight[:1])
            nn.init.xavier_uniform_(linear.weight[1:], gain=relu_gain)
        else:
            nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
            
        nn.init.zeros_(linear.bias)
    def compute_raw_density(
        self, 
        opt: Dict[str, Any],
        H: int, 
        W: int,
        depth_range: List[int],
        intr: torch.Tensor,
        points_3D_samples: torch.Tensor,
        embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor]
    ) -> torch.Tensor:
        """Compute raw density values from 3D points using feature volume.
        
        Args:
            opt: Configuration options
            H: Image height
            W: Image width
            depth_range: [min_depth, max_depth] range
            intr: Camera intrinsic matrix [B, 3, 3]
            points_3D_samples: 3D points to query [B, N_rays, N_samples, 3]
            embedder_pts: Positional encoding function for points
            
        Returns:
            Raw density values for input points [B, N_rays, N_samples, C]
        """
        # Get feature volume dimensions
        _, Cdim, Ddim, Hdim, Wdim = self.feature_volume.shape
        depth_min, depth_max = depth_range

        # Project 3D points to 2D
        points_2D = points_3D_samples @ intr.unsqueeze(1).transpose(-1, -2)
        points_2Dx, points_2Dy, points_2Dz = torch.split(points_2D, 1, dim=-1)

        # Normalize coordinates to [-1, 1] range
        points_2Dx = (points_2Dx / (points_2Dz + 1e-10) / W - 0.5) * 2
        points_2Dy = (points_2Dy / (points_2Dz + 1e-10) / H - 0.5) * 2
        
        # Verify depth parameterization
        assert self.opt.nerf.depth.param == 'datainverse', "Only datainverse depth parameterization implemented"
        
        # Convert depth to normalized range
        points_2Dz = ((1 / points_2Dz - depth_min) / (depth_max - depth_min) - 0.5) * 2

        # Apply activation to feature volume
        if self.unique_3D_act:
            density = torch.nn.functional.softmax(
                self.feature_volume * self.unique_3D_act_temperature, dim=2
            ) * (torch.relu(self.learnable_max_value) + 0.01)
        else:
            density_activ = getattr(torch_F, opt.arch.density_activ)
            density = density_activ(self.feature_volume)

        # Sample density at projected points
        vgrid = torch.cat([points_2Dx, points_2Dy, points_2Dz], dim=-1)
        b, nray, npts, _ = vgrid.shape
        
        # Reshape for grid sampling
        vgrid = einops.rearrange(vgrid, 'b nr np nd -> 1 (b nr np) 1 1 nd')
        
        # Sample features using grid sampling
        points_3D_feature = torch.nn.functional.grid_sample(
            density, 
            vgrid,
            padding_mode='border',
            align_corners=True,
            mode='bilinear'
        ).squeeze(0).squeeze(-1).squeeze(-1)
        
        # Reshape to original dimensions
        points_3D_feature = einops.rearrange(
            points_3D_feature, 
            'cdim (b nr np) -> cdim b nr np', 
            cdim=Cdim, b=b, nr=nray, np=npts
        )
        points_3D_feature = torch.permute(points_3D_feature, [1, 2, 3, 0])
        
        return points_3D_feature

    def forward(
        self,
        opt: Dict[str, Any],
        H: int,
        W: int,
        depth_range: List[int],
        intr: torch.Tensor,
        points_3D_samples: torch.Tensor,
        ray: torch.Tensor,
        embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
        embedder_view: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate predictions for sampled 3D points along rays.
        
        Computes density and RGB values for points sampled along camera rays.
        
        Args:
            opt: Configuration options
            H: Image height
            W: Image width
            depth_range: [min_depth, max_depth] range
            intr: Camera intrinsic matrix [B, 3, 3]
            points_3D_samples: Sampled 3D points [B, N_rays, N_samples, 3]
            ray: Ray directions [B, N_rays, 3]
            embedder_pts: Positional encoding function for points
            embedder_view: Positional encoding function for viewing directions
            mode: Optional mode flag for different forward behaviors
            
        Returns:
            Dictionary containing:
            - rgb_samples: RGB values for sampled points [B, N_rays, N_samples, 3]
            - density_samples: Density values for sampled points [B, N_rays, N_samples]
        """
        # Compute raw density values from feature volume
        density = self.compute_raw_density(
            opt, H, W, depth_range, intr, points_3D_samples, embedder_pts
        )

        # Package predictions
        pred = {
            # Initialize RGB values as zeros
            'rgb_samples': torch.zeros_like(density).repeat([1, 1, 1, 3]),
            # Remove singleton dimension from density
            'density_samples': density.squeeze(-1)
        }
        
        return pred


    def positional_encoding(
        self,
        opt: Dict[str, Any],
        input: torch.Tensor,
        embedder_fn: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
        L: int
    ) -> torch.Tensor:
        """Apply coarse-to-fine positional encoding following BARF strategy.
        
        Implements progressive frequency band activation for positional encoding,
        allowing the model to learn coarse features first before fine details.
        
        Args:
            opt: Configuration options
            input: Input tensor to encode [B, ..., C]
            embedder_fn: Function that applies positional encoding
            L: Number of frequency bands
            
        Returns:
            Encoded tensor with shape [B, ..., 2NL] where N is input channels
        """
        # Apply base positional encoding
        input_enc = embedder_fn(opt, input, L)
        
        # Apply coarse-to-fine masking if enabled
        if opt.barf_c2f is not None:
            # Calculate progress through training
            start, end = opt.barf_c2f
            alpha = (self.progress.data - start) / (end - start) * L
            
            # Generate frequency band weights
            k = torch.arange(L, dtype=torch.float32, device=input.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            
            # Apply progressive masking
            input_enc = (
                input_enc.view(-1, L) * weight
            ).view(*input_enc.shape)
            
        return input_enc

    def forward_samples(
        self,
        opt: Dict[str, Any],
        center: torch.Tensor,
        ray: torch.Tensor,
        depth_samples: torch.Tensor,
        intr: torch.Tensor,
        H: int,
        W: int,
        depth_range: List[float],
        embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
        embedder_view: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate predictions for points sampled along camera rays.
        
        This method:
        1. Converts depth samples to 3D points using camera parameters
        2. Passes points through the network to get density and color predictions
        
        Args:
            opt: Configuration options
            center: Camera centers [B, N, 3]
            ray: Ray directions [B, N, 3]
            depth_samples: Sampled depths along rays [B, N, N_samples, 1]
            intr: Camera intrinsic matrix [B, 3, 3]
            H: Image height
            W: Image width
            depth_range: [min_depth, max_depth] range
            embedder_pts: Positional encoding function for points
            embedder_view: Positional encoding function for viewing directions
            mode: Optional mode flag for different behaviors
            
        Returns:
            Dictionary containing network predictions for sampled points
        """
        # Convert depth samples to 3D points
        points_3D_samples = camera.get_3D_points_from_depth(
            center, ray, depth_samples, multi_samples=True
        )
        
        # Get network predictions for points
        pred_dict = self.forward(
            opt=opt,
            H=H,
            W=W, 
            depth_range=depth_range,
            intr=intr,
            points_3D_samples=points_3D_samples,
            ray=ray,
            embedder_pts=embedder_pts,
            embedder_view=embedder_view,
            mode=mode
        )
        
        return pred_dict

    def composite(
        self,
        opt: Dict[str, Any],
        ray: torch.Tensor,
        pred_dict: Dict[str, Any],
        depth_samples: torch.Tensor
    ) -> Dict[str, Any]:
        """Transform raw network predictions into final rendered outputs.
        
        Implements volume rendering by:
        1. Computing transmittance and alpha values from density
        2. Using these to weight color and depth samples
        3. Integrating along rays to get final RGB and depth values
        
        Args:
            opt: Configuration options
            ray: Ray directions [B, num_rays, 3]
            pred_dict: Network predictions containing:
                - rgb_samples: RGB values [B, num_rays, N_samples, 3]
                - density_samples: Density values [B, num_rays, N_samples]
            depth_samples: Sampled depths along rays [B, num_rays, N_samples, 1]
            
        Returns:
            Updated prediction dictionary containing:
            - rgb: Final RGB colors [B, num_rays, 3]
            - depth: Rendered depth map [B, num_rays, 1]
            - opacity: Ray termination probabilities [B, num_rays, 1]
            - weights: Sample weights [B, num_rays, N_samples, 1]
            - Additional variance and transmittance values
        """
        # Extract predictions
        rgb_samples = pred_dict['rgb_samples']
        density_samples = pred_dict['density_samples']
        
        # Get ray lengths for distance calculations
        ray_length = ray.norm(dim=-1, keepdim=True)

        # Compute distances between samples
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]
        depth_intv_samples = torch.cat([
            depth_intv_samples,
            torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)
        ], dim=2)
        dist_samples = depth_intv_samples * ray_length

        # Compute alpha compositing weights
        sigma_delta = density_samples * dist_samples
        alpha = 1 - (-sigma_delta).exp_()
        
        # Compute transmittance (probability of ray reaching each sample)
        T = (-torch.cat([
            torch.zeros_like(sigma_delta[..., :1]),
            sigma_delta[..., :-1]
        ], dim=2).cumsum(dim=2)).exp_()
        
        # Store cumulative transmittance
        all_cumulated = T[:, :, -2].clone()

        # Compute final weights for integration
        weights = (T * alpha)[..., None]

        # Integrate weighted colors and depths
        depth = (depth_samples * weights).sum(dim=2)
        depth_var = (weights * (depth_samples - depth.unsqueeze(-1))**2).sum(dim=2)

        rgb = (rgb_samples * weights).sum(dim=2)
        rgb_var = ((rgb_samples - rgb.unsqueeze(-2)).sum(dim=-1, keepdims=True) * weights).sum(dim=2)

        # Compute opacity (probability of ray termination)
        opacity = weights.sum(dim=2)

        # Handle background color
        if opt.nerf.setbg_opaque or opt.mask_img:
            rgb = rgb + (1 - opacity)

        # Update predictions with rendered outputs
        pred_dict.update({
            'rgb': rgb,
            'rgb_var': rgb_var,
            'depth': depth,
            'depth_var': depth_var,
            'opacity': opacity,
            'weights': weights,
            'all_cumulated': all_cumulated,
            'T': T
        })
        
        return pred_dict
