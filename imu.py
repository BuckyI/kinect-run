"""
export raw IMU data from Kinect video
"""

import csv

import pykinect_azure as pykinect

pykinect.initialize_libraries()
playback = pykinect.start_playback("data/kinect_0107.mkv")
writer = csv.writer(open("kinect_0107_imu.csv", "w", newline=""))
writer.writerow(
    [
        "acc_timestamp_usec",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_timestamp_usec",
        "gyro_raw",
        "gyro_pitch",
        "gyro_yaw",
    ]
)

try:
    while True:
        imu = playback.get_next_imu_sample()
        sample = [
            imu.get_acc_time(),
            *imu.get_acc(),
            imu.get_gyro_time(),
            *imu.get_gyro(),
        ]
        writer.writerow(sample)
except Exception:
    print("finished")
