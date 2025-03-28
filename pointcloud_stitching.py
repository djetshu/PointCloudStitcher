import argparse
import os
import torch
import numpy as np
torch.cuda.empty_cache()
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
import open3d as o3d

from experiments.geotransformer_3dmatch.config import make_cfg
from experiments.geotransformer_3dmatch.model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="path to the data folder")
    parser.add_argument("--output-path", required=False, help="output path")
    parser.add_argument("--show", action="store_true", help="show the pointclouds")
    parser.add_argument("--gt-file", required=False, help="ground-truth transformation file")
    parser.add_argument("--weights", required=False, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
    else:
        transform = np.eye(4)  # 4x4 identity matrix

    data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def load_data_it(src_path, ref_path, args):
    src_points = np.load(src_path)
    ref_points = np.load(ref_path)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
    else:
        transform = np.eye(4)  # 4x4 identity matrix

    data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    input_src = sorted([os.path.join(args.data_path, npy_file) for npy_file in os.listdir(args.data_path) if npy_file.endswith('.npy')])
    
    ply_files = ['./data/experiments/exp/ply/rgb_points_20250303_134004.ply',
                 './data/experiments/exp/ply/rgb_points_20250303_134039.ply',
                 './data/experiments/exp/ply/rgb_points_20250303_134111.ply',
                 './data/experiments/exp/ply/rgb_points_20250303_134146.ply',]
    
    estimated_transform_list = []
    ref_points_list = []
    src_points_list = []
    color_list = ["custom_yellow",
                  "custom_blue",
                  "custom_red",
                  "custom_green",
                  "custom_orange",
                  "custom_purple",
                  "custom_cyan",
                  "custom_pink",
                  "custom_gray",
                  "custom_brown",]
    
    WEIGHTS_PATH = './weights/geotransformer-3dmatch.pth.tar'

    for i in range(len(input_src)-1):
        # prepare data
        data_dict = load_data_it(input_src[i], input_src[i+1], args)
        neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )

        # prepare model
        model = create_model(cfg).cuda()
        state_dict = torch.load(WEIGHTS_PATH)
        model.load_state_dict(state_dict["model"])

        with torch.no_grad():
            model.eval()
            # prediction
            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

            # get results
            ref_points = output_dict["ref_points"]
            src_points = output_dict["src_points"]
            estimated_transform = output_dict["estimated_transform"]
            transform_gt = data_dict["transform"]

        estimated_transform_list.append(estimated_transform)
        ref_points_list.append(ref_points)
        src_points_list.append(src_points)

    # visualization
    point_clouds = []
    
    # Initialize an empty point cloud
    stitched_pcd = o3d.geometry.PointCloud()
    stitched_rgb_pcd = o3d.geometry.PointCloud()

    for i, (ref_points, src_points) in enumerate(zip(ref_points_list, src_points_list)):
        ref_pcd = make_open3d_point_cloud(ref_points)
        ref_pcd.estimate_normals()
        ref_pcd.paint_uniform_color(get_color(color_list[i + 1]))
        ref_rgb_pcd = o3d.io.read_point_cloud(ply_files[i + 1])
        for transform in estimated_transform_list[i+1:]:
            ref_pcd.transform(transform)
            transform_m = transform.copy()
            transform_m[:3, 3] *= 1000
            ref_rgb_pcd.transform(transform_m)
        point_clouds.append(ref_pcd)
        
        src_pcd = make_open3d_point_cloud(src_points)
        src_pcd.estimate_normals()
        src_pcd.paint_uniform_color(get_color(color_list[i]))
        src_rgb_pcd = o3d.io.read_point_cloud(ply_files[i])
        for transform in estimated_transform_list[i:]:
            src_pcd.transform(transform)
            transform_m = transform.copy()
            transform_m[:3, 3] *= 1000
            src_rgb_pcd.transform(transform_m)
        point_clouds.append(src_pcd)

        # Merge into the final stitched point cloud
        stitched_pcd += ref_pcd
        stitched_pcd += src_pcd

        stitched_rgb_pcd += ref_rgb_pcd
        stitched_rgb_pcd += src_rgb_pcd
    
    draw_geometries(*point_clouds)

    # Visualize the final stitched point cloud
    o3d.visualization.draw_geometries([stitched_pcd])
    o3d.visualization.draw_geometries([stitched_rgb_pcd])

    o3d.io.write_point_cloud("./data/ply/exp/stitched.ply", stitched_pcd)
    print("Final stitched point cloud saved as 'stitched.ply'")
    
    # Compute error
    if args.gt_file is not None:
        rre, rte = compute_registration_error(transform_gt, estimated_transform_list[0])
        print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
