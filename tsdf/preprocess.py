"mkv -> depth, color"
import argparse
import json
from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import numpy as np
import pykinect_azure as pykinect


class Frame(NamedTuple):
    depth: np.ndarray
    color: np.ndarray


def frame_from_video(filename: str) -> Generator[Frame, None, None]:
    """
    filename: mkv 文件
    """
    pykinect.initialize_libraries()
    playback = pykinect.start_playback(filename)
    while True:
        res, capture = playback.update()
        if not res:
            print("finished")
            break

        res1, depth = capture.get_transformed_depth_image()  # type: ignore
        res2, color = capture.get_color_image()  # type: ignore
        if not res1 or not res2:
            print("unexpexted frame")
            continue

        yield Frame(depth, color)  # type: ignore


def intrinsic(filename: str) -> dict:
    pykinect.initialize_libraries()
    playback = pykinect.start_playback(filename)
    calibration = playback.get_calibration()
    # see: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/calibration/main.cpp#L79-L80
    return {
        "width": calibration._handle.color_camera_calibration.resolution_width,  # type: ignore
        "height": calibration._handle.color_camera_calibration.resolution_height,  # type: ignore
        "intrinsic_params": {
            "cx": calibration.color_params.cx,
            "cy": calibration.color_params.cy,
            "fx": calibration.color_params.fx,
            "fy": calibration.color_params.fy,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("output", type=str, default="output")
    args = parser.parse_args()

    assert Path(args.filename).exists(), f"{args.filename} not exists, abort"
    # create output dir
    output_dir = Path(args.output)
    assert not output_dir.exists(), f"{output_dir} already exists, abort"
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    for d in [color_dir, depth_dir]:
        d.mkdir(parents=True)

    # output
    json.dump(intrinsic(args.filename), open(output_dir / "intrinsic.json", "w"))
    for idx, frame in enumerate(frame_from_video(args.filename)):
        filename = f"{idx:04}.png"
        cv2.imwrite(str(color_dir / filename), frame.color)
        cv2.imwrite(str(depth_dir / filename), frame.depth)
