# PointCloudStitcher

## Overview
PointCloudStitcher is a tool designed to stitch multiple `.ply` files into a single `.ply` file. It utilizes a high-performance point cloud registration algorithm, optimized for partial overlaps of 10–30%.

## Requirements
### Data
- `.ply` files must have at least 30% partial overlap. A higher overlap percentage generally results in a more accurately stitched final file.
- Files should be named in ascending or alphabetical order, ensuring that each file overlaps with the next one in sequence.
- Files should be located in `./data/experiments/exp/ply`, relative to the repository directory.

### Software
- **CUDA**: Version 11.8 (Tested)
- **cuDNN**: Version > 8.9 (Tested – Other versions may also work)

## Installation
Follow these steps to install the software:

```bash
# Create a new virtual environment (recommended)
conda create -n pointcloud_stitcher python==3.8
conda activate pointcloud_stitcher

# Clone the repository
git clone git@github.com:djetshu/PointCloudStitcher.git
cd PointCloudStitcher

# Install PyTorch (Ensure compatibility with CUDA 11.8)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt

# Build the package
python setup.py build develop
```

## Pre-trained Weights
Download the `geotransformer-3dmatch.pth.tar` weight file from the [release page](https://github.com/qinzheng93/GeoTransformer/releases).

After downloading, place the weights in the `weights/` directory.

## Quick Test
### Steps:
1. Ensure `.ply` files are located in `./data/experiments/exp/ply`. If the `exp` or `ply` directories do not exist, create them.
2. Run the following command:

```bash
python pointcloud_stitching.py --data-path data/experiments/exp
```

### Arguments:
- `--data-path`: Path to the experiment directory (default: `data/experiments/exp`).
- `--weights`: Path to the pre-trained model weights (default: `./weights/geotransformer-3dmatch.pth.tar`).
- `--no-graph`: Disables graphical visualization of the stitching process.
- `--output`: Directory for saving the final `.ply` file (default: `data/experiments/exp/output`).

3. Once processing is complete, the final `.ply` file will be located in `data/experiments/exp/output`.

## Data Organization
The dataset should be structured as follows:
```text
--data
   |--experiments
       |--exp
           |--ply
               |--rgb_0.ply
               |--rgb_1.ply
               |--rgb_2.ply
               |--rgb_3.ply
           |--npy (Created by the program)
           |--output (Created by the program)
```

## Geometric Transformer for Fast and Robust Point Cloud Registration
PointCloudStitcher is based on [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), an implementation of the paper:

[Geometric Transformer for Fast and Robust Point Cloud Registration](https://arxiv.org/abs/2202.06688)

### Tested Environment:
- **OS**: Ubuntu 22.04
- **Python**: 3.8
- **PyTorch**: 1.7.1
- **CUDA**: 11.8
- **cuDNN**: 8.9

