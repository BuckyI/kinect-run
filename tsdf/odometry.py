import copy
import json
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import open3d as o3d
from tqdm import tqdm


class ImagePair(NamedTuple):
    source_color: str
    source_depth: str
    target_color: str
    target_depth: str


def image_pairs(database_path: Path) -> list[ImagePair]:
    """Find image pairs in the database."""
    colors = sorted(database_path.glob("color/*.png"))
    depths = sorted(database_path.glob("depth/*.png"))
    assert len(colors) == len(depths)
    results = []
    for sc, sd, tc, td in zip(colors, depths, colors[1:], depths[1:]):
        results.append(ImagePair(str(sc), str(sd), str(tc), str(td)))
    return results


def get_translation(image_pair: ImagePair, visualize: bool = False) -> np.ndarray:
    """Find camera movement between two consecutive RGBD image pairs"""

    source_color = o3d.io.read_image(image_pair.source_color)
    source_depth = o3d.io.read_image(image_pair.source_depth)
    target_color = o3d.io.read_image(image_pair.target_color)
    target_depth = o3d.io.read_image(image_pair.target_depth)

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth
    )
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth
    )
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd_image, intrinsic
    )
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, intrinsic
    )

    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    [success, translation, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        intrinsic,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    assert success  # TODO
    if visualize:
        source_pcd = copy.deepcopy(source_pcd)
        source_pcd.transform(translation)
        o3d.visualization.draw([target_pcd, source_pcd])

    return translation


if __name__ == "__main__":
    dataset_path = Path(r"E:\kinect-run\data\test")

    # load intrinsic
    params = json.load(open(dataset_path / "intrinsic.json"))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        params["width"],
        params["height"],
        params["intrinsic_params"]["fx"],
        params["intrinsic_params"]["fy"],
        params["intrinsic_params"]["cx"],
        params["intrinsic_params"]["cy"],
    )

    poses = [np.eye(4)]
    for pair in tqdm(image_pairs(dataset_path)):
        source_color = o3d.io.read_image(pair.source_color)
        trans = get_translation(pair)
        poses.append(trans @ poses[-1])

    with open(dataset_path / "odometry.pickle", "wb") as f:
        pickle.dump(poses, f)
