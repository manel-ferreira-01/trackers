import numpy as np
from scipy.optimize import least_squares
import cv2
import torch 

# ─── Projection & Residuals ───────────────────────────────────────────────────

def project_points(points_3d, rvec, tvec, K):
    R, _ = cv2.Rodrigues(rvec)
    pts  = (R @ points_3d.T + tvec.reshape(3, 1))
    pts /= pts[2]
    return (K @ pts)[:2].T

def bundle_adjustment_residuals(params, points_2d, K, n_cameras, n_points):
    cameras = params[:n_cameras * 6].reshape(n_cameras, 6)
    pts3d   = params[n_cameras * 6:].reshape(n_points, 3)
    residuals = []
    for cam, pts2d in zip(cameras, points_2d):
        projected = project_points(pts3d, cam[:3], cam[3:6], K)
        residuals.append((projected - pts2d).ravel())
    return np.concatenate(residuals)

# ─── Delta Analysis ───────────────────────────────────────────────────────────

def rotation_delta_degrees(rvecs):
    deltas = []
    for i in range(len(rvecs) - 1):
        R1, _ = cv2.Rodrigues(rvecs[i])
        R2, _ = cv2.Rodrigues(rvecs[i + 1])
        R_delta   = R2 @ R1.T
        cos_angle = np.clip((np.trace(R_delta) - 1) / 2, -1, 1)
        deltas.append(np.degrees(np.arccos(cos_angle)))
    return np.array(deltas)

def translation_delta(tvecs):
    return np.linalg.norm(np.diff(tvecs, axis=0), axis=1)

def print_ba_comparison(rvecs_before, tvecs_before, rvecs_after, tvecs_after):
    drb = rotation_delta_degrees(rvecs_before)
    dra = rotation_delta_degrees(rvecs_after)
    dtb = translation_delta(tvecs_before)
    dta = translation_delta(tvecs_after)

    print(f"\n{'':>20}  {'BEFORE':>10}  {'AFTER':>10}  {'DELTA':>10}")
    print("-" * 55)
    print(f"{'Mean rot (°)':>20}  {drb.mean():>10.3f}  {dra.mean():>10.3f}  {dra.mean()-drb.mean():>+10.3f}")
    print(f"{'Max rot (°)':>20}  {drb.max():>10.3f}  {dra.max():>10.3f}  {dra.max()-drb.max():>+10.3f}")
    print(f"{'Mean trans':>20}  {dtb.mean():>10.4f}  {dta.mean():>10.4f}  {dta.mean()-dtb.mean():>+10.4f}")
    print(f"{'Max trans':>20}  {dtb.max():>10.4f}  {dta.max():>10.4f}  {dta.max()-dtb.max():>+10.4f}")

    print(f"\n{'Frame':>6}  {'rot before':>10}  {'rot after':>10}  {'change':>10}")
    print("-" * 42)
    for i, (rb, ra) in enumerate(zip(drb, dra)):
        flag = " <<<" if abs(ra - rb) > 1.0 else ""
        print(f"{i:>3}→{i+1:<3}  {rb:>10.3f}  {ra:>10.3f}  {ra-rb:>+10.3f}{flag}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run_bundle_adjustment(rotations, translations, points_3d, points_2d, K):
    n_cameras = len(rotations)
    n_points  = points_3d.shape[0]

    # Convert rotation matrices to rvecs if needed
    rvecs = []
    for R in rotations:
        if np.array(R).shape == (3, 3):
            rvec, _ = cv2.Rodrigues(np.array(R))
            rvecs.append(rvec.ravel())
        else:
            rvecs.append(np.array(R).ravel())

    rvecs_init = np.array(rvecs)
    tvecs_init = np.array(translations)

    camera_params = np.hstack([rvecs_init, tvecs_init]).ravel()
    x0 = np.concatenate([camera_params, points_3d.ravel()])

    result = least_squares(
        bundle_adjustment_residuals,
        x0,
        method='trf',
        args=(points_2d, K, n_cameras, n_points),
        verbose=2,
    )

    refined_cameras = result.x[:n_cameras * 6].reshape(n_cameras, 6)
    refined_points  = result.x[n_cameras * 6:].reshape(n_points, 3)
    refined_rvecs   = refined_cameras[:, :3]
    refined_tvecs   = refined_cameras[:, 3:6]

    # Before/after comparison
    print_ba_comparison(rvecs_init, tvecs_init, refined_rvecs, refined_tvecs)
    print(f"\nFinal cost: {result.cost:.4f}")

    return refined_rvecs, refined_tvecs, refined_points, result


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

