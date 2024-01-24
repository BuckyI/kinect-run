"""
Basic operations based on sensor SDK tutorial
https://learn.microsoft.com/zh-cn/azure/kinect-dk/about-sensor-sdk
Note some code utilize pykinect_azure's wrapper instead of low level SDK
"""
import cv2
import matplotlib.pyplot as plt
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a


def device_operation():
    # Discover the number of connected devices
    device_count = pykinect.Device.device_get_installed_count()
    print("Found %s connected devices:" % device_count)

    for device_index in range(device_count):
        # Open a device
        device = pykinect.Device(device_index)

        # Read the serial number from the device
        print("{}: Device `{}`".format(device_index, device.get_serialnum()))

        # Close the device
        device.close()


def image_data_operation():
    # Configurations
    device_config = pykinect.default_configuration
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    # Get a capture from the device
    capture = device.update()

    depth_image_object = capture.get_depth_image_object()
    print(
        " | Depth16 res:{}x{} stride:{}".format(
            depth_image_object.get_height_pixels(),
            depth_image_object.get_width_pixels(),
            depth_image_object.get_stride_bytes(),
        )
    )

    ret, depth_image = depth_image_object.to_numpy()  # capture.get_depth_image()
    if not ret:
        return  # fail

    plt.imshow(depth_image, cmap="plasma")
    plt.colorbar()
    plt.show()


def imu_data_operation():
    # Configurations
    device_config = pykinect.default_configuration
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    print(device_config)

    device = pykinect.start_device(config=device_config)
    imu_sample = device.update_imu()
    print(
        " | Accelerometer temperature:{} x:{} y:{} z: {}".format(
            imu_sample.get_temp(), *imu_sample.get_acc()
        )
    )


def image_transform_operation():
    # Configurations
    device_config = pykinect.default_configuration
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32  # required
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED

    # transform images between coordinated camera system to produce RGB-D images
    device = pykinect.start_device(config=device_config)

    while True:  # color_image_to_depth_camera
        capture = device.update()
        ret, color_image = capture.get_transformed_color_image()
        if not ret:  # first time will fail somehow, so try util `ret` is True
            continue
        color_image[:, :, :3] = color_image[:, :, 2::-1]  # BGR to RGB
        plt.imshow(color_image)
        plt.axis("off")
        plt.show()
        break

    while True:  # color_image_to_depth_camera
        capture = device.update()
        ret, depth_image = capture.get_transformed_depth_image()
        if not ret:
            continue
        plt.imshow(depth_image, cmap="plasma")
        plt.colorbar()
        plt.show()
        break


def playback_operation():
    video_filename = "output.mkv"  # video you have recorded using record tool
    playback = pykinect.start_playback(video_filename)

    playback_config = playback.get_record_configuration()
    print(playback_config)

    # get image at 1s
    playback.seek_timestamp(1000)  # timestamps in microseconds
    ret, capture = playback.update()
    ret_depth, depth_color_image = capture.get_colored_depth_image()
    plt.imshow(depth_color_image)
    plt.show()

    # playback
    playback.seek_timestamp(0)  # return to beginning
    cv2.namedWindow("Play Back", cv2.WINDOW_NORMAL)
    while True:
        ret, capture = playback.update()
        if not ret:
            break
        # Get the colored depth
        ret, color_image = capture.get_color_image()
        if not ret:
            continue
        # Plot the image
        cv2.imshow("Play Back", color_image)

        # Press q key to stop
        if cv2.waitKey(30) == ord("q"):
            break


if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    device_operation()
    image_data_operation()
    imu_data_operation()
    image_transform_operation()
    playback_operation()
