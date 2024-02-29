"""
深度图转点云
"""

import numpy as np
import open3d as o3d
import pykinect_azure as pykinect


def get_depth_calibration(video: str):
    "从录制的 mkv 视频中获取深度图的深度相机标定参数"
    pykinect.initialize_libraries()
    playback = pykinect.start_playback(video)
    calibration = playback.get_calibration()

    # see: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/calibration/main.cpp#L79-L80
    return {
        "width": calibration._handle.depth_camera_calibration.resolution_width,
        "height": calibration._handle.depth_camera_calibration.resolution_height,
        "intrinsic_params": calibration.depth_params,
    }


def depth2pcd(depth_image: str, width: int, height: int, intrinsic_params):
    depth_image = o3d.io.read_image(depth_image)
    # 创建深度相机内参对象
    depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsic_params.fx,
        intrinsic_params.fy,
        intrinsic_params.cx,
        intrinsic_params.cy,
    )
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, depth_intrinsic)
    return pcd


if __name__ == "__main__":
    depth_image = r"assets\20240228171418_depth_raw.png"
    video = r"data\kinect_0107.mkv"  # 视频文件没在 Git 里，可以使用 calibration.pickle instead

    calibration = get_depth_calibration(video)
    width, height, intrinsic = (
        calibration["width"],
        calibration["height"],
        calibration["intrinsic_params"],
    )

    pcd = depth2pcd(depth_image, width, height, intrinsic)

    o3d.visualization.draw_geometries(
        [pcd],
        width=width,
        height=height,
        lookat=np.array([0, 0, 0]),
        up=np.array([0, -1, -0]),
        front=np.array([0, 0, -1]),
        zoom=0.5,
    )
