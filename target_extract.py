"""
点云分割与目标提取
基于聚类方法
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud(r"assets\20240229153411_point_cloud.pcd")
pcd = pcd.voxel_down_sample(voxel_size=0.1)  # downsample
visparam = {
    "lookat": np.array([0, 0, 0]),
    "up": np.array([0, -1, -0]),
    "front": np.array([0, 0, -1]),
    "zoom": 0.5,
}
o3d.visualization.draw_geometries([pcd], **visparam)

# detect & remove the ground using ransac
plane_model, inliers = pcd.segment_plane(
    distance_threshold=10, ransac_n=5, num_iterations=1000
)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], **visparam)
pcd = outlier_cloud

# dbscan clustering
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=20, min_points=30, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd], **visparam)
