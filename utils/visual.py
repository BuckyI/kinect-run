import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualize(geometries: list[o3d.geometry.Geometry], *, title="Open3D"):
    # transform to camera coordinate
    visparam = {
        "lookat": np.array([0, 0, 0]),
        "up": np.array([0, -1, -0]),
        "front": np.array([0, 0, -1]),
        "zoom": 0.5,
    }
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(geometries + [frame], title, **visparam)


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


def visualize_transformation(
    poses: list, positions: list | None = None, fast: bool = True
):
    """
    poses: list of 3x3 matrix
    positions: list of 3x1 matrix
    visualize change of transformation
    """

    def set_view(vis):
        vis_ctrl = vis.get_view_control()
        vis_ctrl.set_front(np.array([0, 0, -1]))
        vis_ctrl.set_lookat(np.array([0, 0, 0]))
        vis_ctrl.set_up(np.array([0, -1, 0]))

    if positions is None:
        positions = [np.zeros((3, 1))] * len(poses)

    if fast:  # step sample to speed up animation
        poses = poses[::10]
        positions = positions[::10]

    transformations = []
    for p in poses:
        t = np.eye(4)
        t[:3, :3] = p
        t[:3, 3] = [0, 0, 0]
        transformations.append(t)

    vis = o3d.visualization.Visualizer()
    vis.create_window("visualize transformation", width=500, height=500)

    # world frame
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    vis.add_geometry(ref)

    # init frame
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])
    ref.transform(transformations[0])
    vis.add_geometry(ref)

    # current frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])
    frame.transform(transformations[0])
    vis.add_geometry(frame)

    set_view(vis)

    for trans in transformations:
        vis.remove_geometry(frame)
        # time.sleep(0.01)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=30, origin=[0, 0, 0]
        )
        frame.transform(trans)

        vis.add_geometry(frame)

        # for some reason, view is reset after adding geometry
        # so make sure to set view again
        set_view(vis)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
