import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


class Dataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.colors = [str(i) for i in sorted(self.path.glob("color/*.png"))]
        self.depths = [str(i) for i in sorted(self.path.glob("depth/*.png"))]
        params = json.load(open(self.path / "intrinsic.json", "r"))
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            params["width"],
            params["height"],
            params["intrinsic_params"]["fx"],
            params["intrinsic_params"]["fy"],
            params["intrinsic_params"]["cx"],
            params["intrinsic_params"]["cy"],
        )

        self.poses = pickle.load(open(self.path / "odometry.pickle", "rb"))
        self.length = len(self.colors)  # size of the dataset
        assert len(self.colors) == len(self.depths) == len(self.poses)


def run_tsdf(dataset_path: str):
    dataset = Dataset(dataset_path)

    size = 1
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=size,
        resolution=512,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=[-size / 2, -size / 2, -size / 2],
    )

    for i in tqdm(range(dataset.length), desc="intergrating tsdf"):
        color = o3d.io.read_image(dataset.colors[i])
        depth = o3d.io.read_image(dataset.depths[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=0.5, convert_rgb_to_intensity=False
        )  # depth_trunc=4.0

        volume.integrate(
            rgbd,
            dataset.intrinsic,
            np.linalg.inv(dataset.poses[i]),
        )

    def visualize(g):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries(
            [g, frame], lookat=[0, 0, 0], up=[0, -1, 1], front=[0, 0, -1], zoom=0.5
        )

    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    visualize(mesh)

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    visualize(voxel_pcd)

    print("Extract voxel-aligned debugging voxel grid")
    voxel_grid = volume.extract_voxel_grid()
    visualize(voxel_grid)

    print("Extract point cloud")
    pcd = volume.extract_point_cloud()
    visualize(pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=str)
    args = parser.parse_args()

    run_tsdf(args.database)
