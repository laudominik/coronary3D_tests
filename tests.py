import numpy as np
import matplotlib.pyplot as plt

from xray_angio_3d import reconstruction, XRayInfo
from vessel_tree_generator.module import *
from distance import icp_error


TREE_PATH="./vessel_tree_generator/RCA_branch_control_points/moderate"
VESSEL_TYPE="RCA"
PIXEL_SPACING = 0.35 / 1000
PROJECTION_PARAMS = {
    "alpha": [0, 90, 0],
    "beta": [0, 0, 90],
    "SID": 1.2,
    "SOD":  0.8,
    "spacing": 0.35 / 1000
}
NUM_PROJECTIONS = 3
NUM_TESTS = 100


def test(do_random = False):
    rng = np.random.default_rng()
    
    gt = ensure_generate_vessel_3d()
    projections = []

    for i in range(NUM_PROJECTIONS):
        projection = make_projection(gt, 
        PROJECTION_PARAMS["alpha"][i], PROJECTION_PARAMS["beta"][i], 
        PROJECTION_PARAMS["SOD"], PROJECTION_PARAMS["SID"],
        (PROJECTION_PARAMS["spacing"], PROJECTION_PARAMS["spacing"]))
        xinf = XRayInfo()
        xinf.width = 512
        xinf.height = 512
        xinf.image = projection
        xinf.acquisition_params = {
            'sid': PROJECTION_PARAMS["SID"],
            'sod': PROJECTION_PARAMS["SOD"],
            'alpha': PROJECTION_PARAMS["alpha"][i],
            'beta': PROJECTION_PARAMS["beta"][i],
            'spacing_r': PROJECTION_PARAMS["spacing"],
            'spacing_c': PROJECTION_PARAMS["spacing"]
        }
   
        projections.append(xinf)
    
    if do_random: y = np.random.rand(500, 3)
    else: y = np.array(reconstruction(projections)['vessel'])

    icp_mse = icp_error(gt, y)
    return icp_mse


def ensure_generate_vessel_3d():
    # vessel can be None due to invalid subsampling
    rng = np.random.default_rng()
    vessel = None
    while vessel is None: vessel, _, _ =  generate_vessel_3d(rng, VESSEL_TYPE, TREE_PATH, True, False)
    return vessel


if __name__ == "__main__":

    icp_mse_random = test(True)
    print(f"metric of random points {icp_mse_random}")

    mses = []
    sum_mse = 0
    for i in range(NUM_TESTS):
        icp_mse = test()
        print(f"{i}-th test MSE {icp_mse}")
        mses.append(icp_mse)

    avg_mse = np.average(mses)
    stdev_mse = np.std(mses)
    print(f"average MSE {avg_mse}")
    print(f"stdev of MSE {stdev_mse}")
