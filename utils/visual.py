import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualize(geometries: list[o3d.geometry.Geometry]):
    # transform to camera coordinate
    visparam = {
        "lookat": np.array([0, 0, 0]),
        "up": np.array([0, -1, -0]),
        "front": np.array([0, 0, -1]),
        "zoom": 0.5,
    }
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(geometries + [frame], **visparam)


def select_points(pcd) -> list[int]:
    """
    1) pick at least one point using [shift + left click]
        Press [shift + right click] to undo point picking
    2) After picking points, press 'Q' to close the window
    return the index of selected points
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)

    # transform to camera coordinate
    vis_ctrl = vis.get_view_control()
    vis_ctrl.set_front(np.array([0, 0, -1]))
    vis_ctrl.set_lookat(np.array([0, 0, 0]))
    vis_ctrl.set_up(np.array([0, -1, 0]))

    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()
