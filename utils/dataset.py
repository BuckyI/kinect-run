import time
from pathlib import Path
from typing import Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect


def point_cloud_from_video(
    filename: str, timestep: int = 500000, voxel_size: float = 1
) -> Generator[o3d.geometry.PointCloud, None, None]:
    """
    从视频中提取点云
    filename: mkv 文件
    timestap: 间隔一定时间取一帧，单位为微秒，默认 0.5s
    voxel_size: 原始点云进行降采样的体素大小，默认为 1(mm)
    """
    pykinect.initialize_libraries()
    playback = pykinect.start_playback(filename)

    timestamp = 0  # time of current frame, in microsecond
    while timestamp < playback.get_recording_length():
        playback.seek_timestamp(timestamp)
        res, capture = playback.update()
        if res and capture is not None:
            res, points = capture.get_pointcloud()
            if res and points is not None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # preprocess
                pcd = pcd.remove_duplicated_points()
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)[0]
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=50,  # 5cm 搜索半径 for kinect point cloud
                        max_nn=5,
                    )
                )
                yield pcd
        timestamp += timestep


def point_cloud_from_folder(
    path: str,
) -> Generator[o3d.geometry.PointCloud, None, None]:
    """
    从文件夹中逐个提取点云
    folder: 文件夹路径
    """
    folder = Path(path)
    assert folder.exists() and folder.is_dir()
    for f in sorted(folder.glob("*.pcd")):
        yield o3d.io.read_point_cloud(str(f))
