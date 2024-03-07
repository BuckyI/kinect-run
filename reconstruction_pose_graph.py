import numpy as np
import open3d as o3d
from tqdm import tqdm

from pcd_registration import (
    draw_registration_result,
    fast_global_registration,
    icp,
    preprocess_point_cloud,
)
from utils.dataset import point_cloud_from_folder, point_cloud_from_video
from utils.visual import visualize


def pairwise_registration(source, target):
    global voxel_size, max_correspondence_distance_fine
    trans = np.identity(4)
    for v in [voxel_size * 15, voxel_size * 5, voxel_size * 1.5]:
        result = icp(
            source.voxel_down_sample(v),
            target.voxel_down_sample(v),
            v,
            trans,
            "point_to_plane",
        )
        trans = result.transformation
        # draw_registration_result(source, target, trans, f"{v}")

    transformation = result.transformation  # type: ignore
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, voxel_size * 1.5, transformation
    )

    # (result.transformation.trace() == 4.0) or
    failed = information[5, 5] / min(len(source.points), len(target.points)) < 0.3  # type: ignore
    return failed, transformation, information


def full_registration(pcds: list[o3d.geometry.PointCloud]):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in tqdm(range(n_pcds), desc="Build PoseGraph"):
        for target_id in tqdm(range(source_id + 1, min(source_id + 6, n_pcds))):
            success, transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id]
            )

            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)  # 累积全局位姿
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=not success,
                    )
                )
            else:  # loop closure case
                if not success:
                    continue
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


if __name__ == "__main__":
    pcds_g = point_cloud_from_video("data/kinect_0107.mkv", 100000, 5)
    pcds = []
    for _ in tqdm(range(20), desc="load point cloud"):
        pcds.append(next(pcds_g))

    voxel_size = 5
    print("Full registration ...")

    # max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=50,  # 0.25,
        preference_loop_closure=2.0,  # 2.0 for fragment registration.
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    print("Transform points and display")
    for point_id in range(len(pcds)):
        print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    visualize(pcds)
