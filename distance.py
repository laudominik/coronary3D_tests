import open3d as o3d
import math
import numpy as np


def icp_error(gt, reconstructed):
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(gt)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(reconstructed)


    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], 
                             [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1,
        max_correspondence_distance=math.inf,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    def distance_measure(pt1, pt2):
        return np.sum((pt1 - pt2)**2)

    pcd1.transform(reg_p2p.transformation)

    corr = reg_p2p.correspondence_set
    return sum([distance_measure(gt[ix_gt], reconstructed[ix_rec]) for (ix_gt, ix_rec) in corr])
    