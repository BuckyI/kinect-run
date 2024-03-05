import time
from typing import Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect


def point_cloud_from_video(
    filename: str, timestep: int = 250000
) -> Generator[o3d.geometry.PointCloud, None, None]:
    """
    从视频中提取点云
    filename: mkv 文件
    timestap: 间隔一定时间取一帧，单位为微秒，默认 0.5s
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
                yield pcd
        timestamp += timestep
