"""
回放 Kinect 录制的 mkv 视频
use matplotlib playback recorded mkv video, and save frames
use keyboard to control, see `on_press` for more info
"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect


class Visualizer:
    def __init__(self, video_filename: str) -> None:
        # init kinect
        pykinect.initialize_libraries()
        self.playback = pykinect.start_playback(video_filename)
        self.playback_config = self.playback.get_record_configuration()
        print(self.playback_config)

        # init plot
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.axes[0].set_axis_off()
        self.axes[1].set_axis_off()
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.update_capture()
        self.plot_capture()
        plt.show()

    def update_capture(self, reverse: bool = False):
        "update frames, reverse means previous frame"
        update = self.playback.get_previous_capture if reverse else self.playback.update
        res, capture = update()
        if res:
            assert capture is not None  # for type checker
            self.capture = capture
        else:
            print("failed to update capture")

    def plot_capture(self):
        "plot frames"
        capture: pykinect.Capture = self.capture
        ret_c, color_image = capture.get_color_image()  # type: ignore
        ret_d, depth_image = capture.get_depth_image()  # type: ignore
        assert ret_c and ret_d, "capture is not valid"
        assert color_image is not None and depth_image is not None  # for type checker
        self._depth_image_raw = depth_image

        # depth2rgb
        # alpha is fitted by visual comparison with Azure k4aviewer results
        depth_image = cv2.convertScaleAbs(depth_image, alpha=0.05)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

        ax1, ax2 = self.axes
        ax1.cla(), ax2.cla()  # type: ignore
        ax1.imshow(color_image[:, :, ::-1])  # bgr2rgb
        ax2.imshow(depth_image[:, :, ::-1])
        self.fig.canvas.draw()

        # cache image
        self._depth_image = depth_image
        self._color_image = color_image

    def show_3d_points(self):
        "展示当前帧的 3D 点云"
        ret, points = self.capture.get_pointcloud()
        assert ret
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries(
            [pcd],
            lookat=np.array([0, 0, 0]),
            up=np.array([0, -1, -0]),
            front=np.array([0, 0, -1]),
            zoom=0.5,
        )

    def save_capture(self):
        "save current frame to local"
        print("Saving capture...")
        time_prefix = time.strftime("%Y%m%d%H%M%S")
        cv2.imwrite(time_prefix + "_depth.png", self._depth_image)
        cv2.imwrite(time_prefix + "_color.png", self._color_image)
        cv2.imwrite(time_prefix + "_depth_raw.png", self._depth_image_raw)
        print("Saving capture Done")

    def on_press(self, event):
        "control with keyboard"
        match event.key:
            case " ":  # switch to next frame
                self.update_capture()
                self.plot_capture()
            case "right":  # fast forward to next frame
                for _ in range(10):
                    self.update_capture()
                self.plot_capture()
            case "left":  # switch to previous frame
                self.update_capture(reverse=True)
                self.plot_capture()
            case "w":
                self.save_capture()
            case "e":
                self.show_3d_points()
            case "q":  # quit
                self.fig.canvas.stop_event_loop()
            case _:
                print(f"Key pressed: {event.key}")


if __name__ == "__main__":
    v = Visualizer("data/kinect_0107.mkv")
