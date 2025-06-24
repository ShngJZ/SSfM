# RSfM: Revisit Self-supervised Depth Estimation with Local Structure-from-Motion

This repository contains the official implementation of the paper:   
**Revisit Self-supervised Depth Estimation with Local Structure-from-Motion**  
Authors: [Shengjie Zhu](https://shngjz.github.io/) and [Xiaoming Liu](https://cvlab.cse.msu.edu/)<br>
*ECCV'24* [📄 [arXiv]](https://arxiv.org/abs/2407.19166), [🌐 [webpage]](https://shngjz.github.io/SSfM.github.io/)


<img src="/assets/rsfm_recombined.gif" width="600" >

This work studies a multiview-RANSAC scheme to find multi-view poses that maximize inliers. We further investigate whether such poses can improve state-of-the-art supervised depth map performance in a self-supervised manner by employing a NeRF as a triangulation machine.

## Installation

### Environment Setup

```bash
git clone https://github.com/ShngJZ/RSfM
cd RSfM
git submodule update --init --recursive --remote

conda create -n rsfm python=3.10
conda activate rsfm

# Install PyTorch (https://pytorch.org/)
# Check and select the version compatible with your CUDA (check your CUDA version with nvcc --version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install packages
pip install -r requirements.txt

# Install cupy according to your CUDA version, for CUDA 12.x:
pip install cupy-cuda12x
pip install timm==0.6.7  # ZoeDepth requirement

# Install LightedDepth
cd third_party/LightedDepth/GPUEPMatrixEstimation/
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')" && nvcc --version | grep 'Cuda compilation tools' # ensure the version of Pytorch and system nccc are the same
python setup.py install


# Download pretrained depth and correspondence checkpoints
mkdir checkpoint && cd checkpoint
gdown 1nOpC0MFWNV8N6ue0csed4I2K_ffX64BL # PDCNet
wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt # ZoeDepth
```

If you are installing a higher CUPY version (e.g., cupy 12x), you need to modify the PDC Net codebase. Update the following files:

1. `third_party/DenseMatching/models/modules/local_correlation/correlation.py`
2. `third_party/DenseMatching/third_party/GOCor/GOCor/local_correlation/correlation.py`

Replace the following code:
```python
# Old code
@cupyutil.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)

# New code
@cupyutil.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    module = cupy.RawModule(code=strKernel, name_expressions=[strFunction])
    return module.get_function(strFunction)
```

### Dataset Setup

Please ensure you agree to all licenses of the dataset before downloading.

```bash
# Download datasets
mkdir RSfM-Datasets && cd RSfM-Datasets

# ScanNet dataset, make sure the git lfs installed
git clone https://huggingface.co/datasets/shngjz/f2509922c5dc01d3c0f9a365327c1de3
mv f2509922c5dc01d3c0f9a365327c1de3 ScanNet
```

Note: You may be prompted to enter your user ID and token multiple times. Please expect delays when downloading large datasets.


## ScanNet Experiments

### 1. Run Multiview Pose Estimation and Self-Supervised Depth Learning with NeRF

The main optimization process consists of two steps:
1. Camera pose estimation
2. Depth map optimization using NeRF

For multi-GPU training (example with 8 GPUs), run one command per Tmux window:

```bash
# Replace /home/ubuntu/disk1/RSfM-Datasets/ScanNet with your dataset path
CUDA_VISIBLE_DEVICES=0 python local_ba_scale_up/optimize_pose_depth.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --data_root /home/ubuntu/disk1/RSfM-Datasets/ScanNet --dataset scannet
# Repeat for CUDA_VISIBLE_DEVICES=1 through 7 on your tmux window
```

### 2. Self-supervised Depth Evaluation

```bash
# Compare to initial Mono-Depth inputs:
python evaluation/eval_nerf_depth.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --dataset scannet
```

### 3. Camera Pose Evaluation

```bash
# Evaluate using GT depth map to compare consistent 3D points
python evaluation/eval_pose_by_3dcons.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --dataset scannet

# Evaluate using GT depth map to compare inlier correspondences
python evaluation/eval_pose_by_corres.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --dataset scannet

# Evaluate using predicted depth map to compare consistent depth maps
python evaluation/eval_pose_by_consisdepth.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --dataset scannet

# Evaluate using GT poses
python evaluation/eval_pose_by_pose.py --train_module joint_pose_nerf_training/scannet_depth_exp --train_name zoedepth_pdcnet --train_sub 5 --dataset scannet
```

## Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{zhu2024revisit,
    title={Revisit Self-supervised Depth Estimation with Local Structure-from-Motion},
    author={Zhu, Shengjie and Liu, Xiaoming},
    journal={ECCV},
    year={2024}
}
```
