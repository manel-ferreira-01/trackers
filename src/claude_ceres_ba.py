import pyceres
import numpy as np
import cv2
from src.claude_ba import print_ba_comparison

class ReprojectionError(pyceres.CostFunction):
    def __init__(self, u_obs, v_obs, K):
        super().__init__()
        self.u_obs = u_obs
        self.v_obs = v_obs
        self.K     = K
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([3, 3, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        rvec = np.array(parameters[0])
        tvec = np.array(parameters[1])
        pt3d = np.array(parameters[2])

        R, _ = cv2.Rodrigues(rvec)
        p    = R @ pt3d + tvec
        z    = p[2]

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        u = fx * p[0] / z + cx
        v = fy * p[1] / z + cy

        residuals[:] = [u - self.u_obs, v - self.v_obs]

        if jacobians is not None:
            # Jacobian w.r.t. point in camera coords (dp/dpt3d)
            # p = R @ pt3d + tvec
            # u = fx * p[0]/p[2] + cx
            # v = fy * p[1]/p[2] + cy
            dp_dpt = R  # (3, 3)

            du_dp = np.array([fx/z, 0, -fx*p[0]/z**2])
            dv_dp = np.array([0, fy/z, -fy*p[1]/z**2])

            if jacobians[2] is not None:
                J_pt = np.vstack([du_dp @ dp_dpt,
                                  dv_dp @ dp_dpt])   # (2, 3)
                jacobians[2][:] = J_pt.ravel()

            # Numerical diff for rvec and tvec (simpler, still fast)
            eps = 1e-6
            for param_idx, param in enumerate([rvec, tvec]):
                if jacobians[param_idx] is not None:
                    J = np.zeros((2, 3))
                    for j in range(3):
                        p_fwd        = param.copy(); p_fwd[j]  += eps
                        p_bwd        = param.copy(); p_bwd[j]  -= eps
                        params_fwd   = [p_fwd  if k == param_idx else parameters[k] for k in range(3)]
                        params_bwd   = [p_bwd  if k == param_idx else parameters[k] for k in range(3)]
                        r_fwd        = np.zeros(2); self.Evaluate(params_fwd, r_fwd, None)
                        r_bwd        = np.zeros(2); self.Evaluate(params_bwd, r_bwd, None)
                        J[:, j]      = (r_fwd - r_bwd) / (2 * eps)
                    jacobians[param_idx][:] = J.ravel()

        return True


def run_bundle_adjustment_ceres(rotations, translations, points_3d, points_2d, K):

    # Convert to rvecs
    rvecs = []
    for R in rotations:
        R = np.array(R)
        if R.shape == (3, 3):
            rvec, _ = cv2.Rodrigues(R)
            rvecs.append(rvec.ravel().copy())
        else:
            rvecs.append(R.ravel().copy())

    rvecs_init = np.array([r.copy() for r in rvecs])
    tvecs_init = np.array(translations, dtype=np.float64).copy()

    rvecs  = [r.astype(np.float64) for r in rvecs]
    tvecs  = [np.array(t, dtype=np.float64).ravel().copy() for t in translations]
    pts3d  = points_3d.astype(np.float64).copy()

    problem = pyceres.Problem()
    cost = ReprojectionError(points_2d[0][0][0], points_2d[0][0][1], K)
    residuals = np.zeros(2)
    params = [rvecs[0], tvecs[0], pts3d[0]]
    cost.Evaluate(params, residuals, None)
    print("Test residual:", residuals)   # should be non-zero

    for cam_idx, (rvec, tvec, pts2d) in enumerate(zip(rvecs, tvecs, points_2d)):
        for pt_idx, obs in enumerate(pts2d):
            cost = ReprojectionError(obs[0], obs[1], K)
            problem.add_residual_block(
                cost,
                pyceres.HuberLoss(1.0),         # robust to outliers
                [rvec, tvec, pts3d[pt_idx]]
            )

    options = pyceres.SolverOptions()
    options.linear_solver_type           = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations           = 100
    options.function_tolerance            = 1e-4   # default may be stopping it early
    options.gradient_tolerance            = 1e-6
    options.parameter_tolerance           = 1e-4
    


    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    #print(summary.brief_report())

    refined_rvecs = np.array(rvecs)
    refined_tvecs = np.array(tvecs)

    print_ba_comparison(rvecs_init, tvecs_init, refined_rvecs, refined_tvecs)

    return refined_rvecs, refined_tvecs, pts3d


def make_cam_lists(rvecs, tvecs):
    """
    rvecs:  (F, 3) numpy - Rodrigues vectors
    tvecs:  (F, 3) numpy - translation vectors
    returns: list of (4, 4) torch tensors, same format as input cam_lists
    """
    import torch
    cam_lists_refined = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = tvec
        
        cam_lists_refined.append(torch.tensor(T[:3,:], dtype=torch.float64))
    
    return cam_lists_refined

