import torch
import numpy as np
import cv2
from micro_bundle_adjustment.api import optimize_calibrated
from micro_bundle_adjustment.api import projection
from torch.func import vmap


# ── helpers ───────────────────────────────────────────────────────────────────

def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → axis-angle (Rodrigues) vector (3,)"""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def axis_angle_to_rotmat(r: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(r)
    return R


def make_cam_lists(rvecs, tvecs):
    """
    rvecs:  (F, 3) numpy
    tvecs:  (F, 3) numpy
    returns: list of (3, 4) torch float64 tensors  [same format as input cam_lists]
    """
    cam_lists_refined = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = tvec
        cam_lists_refined.append(torch.tensor(T[:3, :], dtype=torch.float64))
    return cam_lists_refined


# ── main entry point ──────────────────────────────────────────────────────────

def run_bundle_adjustment_micro(
    rotations,       # list of (3,3) or (3,) numpy arrays (R or rvec)
    translations,    # list of (3,) numpy arrays
    points_3d,       # (N, 3) numpy float64
    points_2d,       # list of F arrays, each (N, 2) numpy  (same point order every frame)
    K,               # (3,3) intrinsics — only used if not identity; ignored for calibrated mode
    device="cuda",
    dtype=torch.float64,
    num_steps=100,
    L_0=1e-2,
):
    """
    Drop-in replacement for run_bundle_adjustment_ceres.
    Returns refined_rvecs (F,3), refined_tvecs (F,3), refined_pts3d (N,3) — all numpy float64.

    Assumes K ~ identity (calibrated / normalised coords).
    If your K is not identity, divide points_2d by [fx, fy] and subtract [cx/fx, cy/fy]
    before passing in, or swap to simple_pinhole_residuals.
    """
    F = len(rotations)
    N = points_3d.shape[0]

    # ── initial camera params: theta = [r(3) | t(3)] per frame ────────────────
    r0_list, t0_list = [], []
    for R, t in zip(rotations, translations):
        R = np.array(R)
        r = rotmat_to_axis_angle(R) if R.shape == (3, 3) else R.ravel()
        r0_list.append(r)
        t0_list.append(np.array(t).ravel())

    r0 = torch.tensor(np.array(r0_list), dtype=dtype, device=device)   # (F, 3)
    t0 = torch.tensor(np.array(t0_list), dtype=dtype, device=device)   # (F, 3)

    # ── 3-D points ─────────────────────────────────────────────────────────────
    X0 = torch.tensor(points_3d, dtype=dtype, device=device)            # (N, 3)

    # ── observations: list of (x_im, inds) one entry per frame ────────────────
    # Every point is visible in every frame (matches your dense-track setup).
    inds = torch.arange(N, device=device)
    observations = []
    for f in range(F):
        x_im = torch.tensor(points_2d[f], dtype=dtype, device=device)   # (N, 2)
        observations.append((x_im, inds))

    # ── run ────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        X_hat, theta_hat = optimize_calibrated(
            X0, r0, t0,
            observations,
            dtype=dtype,
            L_0=L_0,
            num_steps=num_steps,
        )

    # ── unpack ─────────────────────────────────────────────────────────────────
    r_hat, t_hat = theta_hat.chunk(2, dim=1)         # (F,3), (F,3)

    refined_rvecs = r_hat.cpu().numpy().astype(np.float64)
    refined_tvecs = t_hat.cpu().numpy().astype(np.float64)
    refined_pts   = X_hat.cpu().numpy().astype(np.float64)

    return refined_rvecs, refined_tvecs, refined_pts