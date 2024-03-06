"""
使用整合点云的思路进行点云重建
通过键盘控制
"""

from pynput import keyboard

from pcd_registration import Model
from utils.dataset import point_cloud_from_video
from utils.visual import visualize


def on_press(key):
    if key == keyboard.KeyCode.from_char("v"):
        print("lets visualize!")
        global vis_flag
        vis_flag = True


vis_flag = False
listener = keyboard.Listener(on_press=on_press)
listener.start()


pcds = (
    p.paint_uniform_color([1, 0.706, 0])
    for p in point_cloud_from_video("data/kinect_0107.mkv", 100000, voxel_size=1)
)

init = next(pcds)
model = Model(init)
visualize([model.pcd], title="init")
for idx, pcd in enumerate(pcds):  # merge
    print("merging", idx)
    model.merge(pcd)
    if vis_flag:
        visualize([model.pcd], title="combination")
        model.pcd.paint_uniform_color([0, 0.651, 0.929])
        vis_flag = False
