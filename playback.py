"""
use matplotlib playback recorded mkv video, and save frames
use keyboard to control, see `on_press` for more info
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
        self.axes[0].set_axis_off(), self.axes[1].set_axis_off()
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.update_capture()
        self.plot_capture()
        plt.show()

    def update_capture(self, reverse: bool = False):
        if reverse:
            res, self.capture = self.playback.get_previous_capture()
        else:
            res, self.capture = self.playback.update()
        if not res:
            print("failed to update capture")

    def plot_capture(self):
        capture: pykinect.Capture = self.capture
        ret_c, color_image = capture.get_color_image()
        ret_d, depth_image = capture.get_depth_image()
        assert ret_c and ret_d, "capture is not valid"

        ax1, ax2 = self.axes
        ax1.cla(), ax2.cla()
        ax1.imshow(color_image[:, :, ::-1])  # bgr2rgb
        ax2.imshow(depth_image, cmap="plasma")
        self.fig.canvas.draw()

    def save_capture(self):
        print("Saving capture...")
        time_prefix = time.strftime("%Y%m%d%H%M%S")
        _, depth = self.capture.get_depth_image()
        depth_color_image = cv2.convertScaleAbs(
            depth, alpha=0.05
        )  # alpha is fitted by visual comparison with Azure k4aviewer results
        depth_color_image = cv2.applyColorMap(depth_color_image, cv2.COLORMAP_JET)
        cv2.imwrite(time_prefix + "_depth.png", depth_color_image)

        _, color_image = self.capture.get_color_image()
        cv2.imwrite(time_prefix + "_color.png", color_image)
        print("Saving capture Done")

    def on_press(self, event):
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
            case "q":  # quit
                self.fig.canvas.stop_event_loop()
            case _:
                print(f"Key pressed: {event.key}")


if __name__ == "__main__":
    v = Visualizer("data/kinect_0107.mkv")
