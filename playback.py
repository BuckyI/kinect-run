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
        self.video_name = video_filename.strip(".mkv")
        pykinect.initialize_libraries()
        self.playback = pykinect.start_playback(video_filename)
        self.playback_config = self.playback.get_record_configuration()
        print(self.playback_config)

        # init control
        self.video_length = self.playback.get_recording_length()  # in microsecond
        self.timestamp = 0  # time of current frame, in microsecond

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
        # update timestamp in 0.5s step
        step: int = -500000 if reverse else 500000
        self.timestamp += step
        if self.timestamp > self.video_length or self.timestamp < 0:
            print("timestamp out of range")
            self.timestamp = min(max(self.timestamp, 0), self.video_length)

        self.playback.seek_timestamp(self.timestamp)
        res, capture = self.playback.update()
        assert res, "failed to update capture"
        assert capture is not None  # for type checker
        self.capture = capture

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
        # set title to current time
        timestr = time.strftime("%H:%M:%S", time.gmtime(self.timestamp / 1e6))
        self.fig.suptitle("frame: " + timestr)

        # cache image
        self._depth_image = depth_image
        self._color_image = color_image

    def get_pointcloud(self):
        ret, points = self.capture.get_pointcloud()
        assert ret
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def show_3d_points(self):
        "展示当前帧的 3D 点云"
        pcd = self.get_pointcloud()
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
        prefix = f"{self.video_name}_{self.timestamp}"
        cv2.imwrite(prefix + "_depth.png", self._depth_image)
        cv2.imwrite(prefix + "_color.png", self._color_image)
        cv2.imwrite(prefix + "_depth_raw.png", self._depth_image_raw)
        o3d.io.write_point_cloud(prefix + "_pointcloud.pcd", self.get_pointcloud())
        print("Saving capture Done")

    def on_press(self, event):
        "control with keyboard"
        match event.key:
            case " ":
                for _ in range(10):  # fast forward to next frame
                    self.update_capture()
                self.plot_capture()
            case "right":  # switch to next frame
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
