import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect

from utils.visual import visualize


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation,
):
    "transformation: 4x4 matrix"
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    visualize([source_temp, target_temp], title="registration result")


def evaluate_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_threshold: float,
) -> o3d.pipelines.registration.RegistrationResult:
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, distance_threshold
    )
    return result


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    "downsample the point cloud, and compute the FPFH feature"
    print(":: Downsample with a voxel size%.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius%.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius%.3f." % radius_feature)
    pcd_fpfh: o3d.pipelines.registration.Feature = (
        o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
    )

    return pcd_down, pcd_fpfh


def global_registration(
    source, source_fpfh, target, target_fpfh, voxel_size
) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,  # Source point cloud feature.
        target_fpfh,  # Target point cloud feature.
        True,  # mutual_filter
        distance_threshold,  # max_correspondence_distance
        # estimation_method
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # ransac_n: Fit ransac with ransac_n correspondences
        # checkers: check if two point clouds can be aligned.
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        # criteria: Convergence criteria
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.999),
    )
    return result


def fast_global_registration(
    source, source_fpfh, target, target_fpfh, voxel_size
) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source,
        target,
        source_fpfh,  # Source point cloud feature.
        target_fpfh,  # Target point cloud feature.
        option=o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    init: np.ndarray,
    icp_method: str = "point_to_point",
) -> o3d.pipelines.registration.RegistrationResult:
    """
    init: Initial transformation 4x4 matrix
    """
    distance_threshold = voxel_size * 0.4

    # compute normals
    without_normals = filter(lambda x: not x.has_normals(), [source, target])
    for p in without_normals:
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

    match icp_method:
        case "point_to_point":
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                distance_threshold,  # max_correspondence_distance
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                init,  # Initial transformation estimation
                # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000)
            )
        case "point_to_plane":
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                distance_threshold,  # max_correspondence_distance
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                init,  # Initial transformation estimation
            )
        case "robust":  # Robust ICP
            sigma = 0.5  # standard deviation of noise
            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                distance_threshold,
                init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            )

        case "generalized":
            result = o3d.pipelines.registration.registration_generalized_icp(
                source,
                target,
                distance_threshold,
                init,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                # o3d.pipelines.registration.ICPConvergenceCriteria(
                #     relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
                # ),
            )
        case _:
            raise ValueError(f"unknown icp_method: {icp_method}")

    return result


def simple_registration_combination():
    pcd_paths = [
        "assets/kinect_0107_2000000_pointcloud.pcd",
        "assets/kinect_0107_2500000_pointcloud.pcd",
        "assets/kinect_0107_3000000_pointcloud.pcd",
        "assets/kinect_0107_3500000_pointcloud.pcd",
    ]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    pcds = [o3d.io.read_point_cloud(path) for path in pcd_paths]
    for i in range(len(pcds)):
        pcds[i] = pcds[i].remove_duplicated_points()
        pcds[i] = pcds[i].paint_uniform_color(colors[i])
        pcds[i] = pcds[i].voxel_down_sample(voxel_size=1)
        pcds[i].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    visualize(pcds, title="load point cloud")

    # prprocess
    voxel_size = 50  # 5cm precision
    preprocessed = [preprocess_point_cloud(pcd, voxel_size) for pcd in pcds]
    visualize([p[0] for p in preprocessed], title="preprocessed point cloud")

    s, t = 0, 1
    # global registration
    result = fast_global_registration(*preprocessed[s], *preprocessed[t], voxel_size)
    print(evaluate_registration(pcds[s], pcds[t], voxel_size))
    print(result.transformation)
    draw_registration_result(pcds[s], pcds[t], result.transformation)

    # local registration
    result = icp(pcds[s], pcds[t], voxel_size, result.transformation, "generalized")
    print(evaluate_registration(pcds[s], pcds[t], voxel_size))
    print(result.transformation)
    draw_registration_result(pcds[s], pcds[t], result.transformation)

    # combination
    p1 = copy.deepcopy(pcds[s])
    p2 = copy.deepcopy(pcds[t])
    result_pcd = p2 + p1.transform(result.transformation)
    result_pcd = result_pcd.voxel_down_sample(voxel_size=1)
    visualize([result_pcd], title="combination")
