import torch
import torch.nn.functional as F

def sample_depths(depths, tracks):
    """
    depths: [F, H, W]  depth maps
    tracks: [2F, P]    pixel coords (u,v)
    Returns: [F, P] sampled depths
    """
    F_, H, W = depths.shape
    P = tracks.shape[1]

    depths = depths.unsqueeze(1)  # [F, 1, H, W]

    all_depths = []
    for f in range(F_):
        u = tracks[2*f, :]   # [P]
        v = tracks[2*f+1, :] # [P]

        # Normalize coords to [-1,1] for grid_sample
        u_norm = (u / (W-1)) * 2 - 1
        v_norm = (v / (H-1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, P, 1, 2)  # [1,P,1,2]

        z = F.grid_sample(
            depths[f:f+1], grid, align_corners=True, mode='bilinear'
        ).view(-1)  # [P]

        all_depths.append(z)

    return torch.stack(all_depths, dim=0)  # [F, P]


def build_depth_weighted_matrix(tracks, depths, fx, fy, cx, cy):
    """
    Build a 3F x P depth-weighted observation matrix.

    Args:
        tracks: torch.Tensor [2F, P]
            Pixel coordinates (u,v) for each feature across frames.
        depths: torch.Tensor [F, H, W]
            Depth map for each frame.
        fx, fy, cx, cy: floats
            Camera intrinsics (assumed same for all frames).

    Returns:
        W_proj: torch.Tensor [3F, P]
            Depth-weighted homogeneous observation matrix.
    """
    F = depths.shape[0]
    P = tracks.shape[1]

    # interpolator of depth at 
    z = sample_depths(depths, tracks) 

    u = tracks[0::2, :]  # [F, P]
    v = tracks[1::2, :]  # [F, P]

    # Normalize to camera plane
    x_norm = (u - cx) / fx  # [F, P]
    y_norm = (v - cy) / fy  # [F, P]

    W_proj = torch.zeros((3 * F, P), dtype=tracks.dtype, device=tracks.device)

    # Fill each frame’s block of 3 rows
    for f in range(F):
        W_proj[3 * f + 0, :] = x_norm[f] * z[f]
        W_proj[3 * f + 1, :] = y_norm[f] * z[f]
        W_proj[3 * f + 2, :] = z[f]

    return W_proj, z


def normalize_measurement_matrix(W):
    """
    Isotropic normalization for 3F x P projective measurement matrix.

    Args:
        W: torch.Tensor [3F, P]
            Depth-weighted homogeneous measurement matrix.

    Returns:
        W_norm: torch.Tensor [3F, P] normalized matrix
        T_list: list of torch.Tensor [4,4]
            Normalization transforms per frame (for upgrading back).
    """
    F = W.shape[0] // 3
    P = W.shape[1]

    W_norm = torch.zeros_like(W)
    T_list = []

    for f in range(F):
        block = W[3*f:3*f+3, :]  # [3, P]

        homog = torch.cat([block, torch.ones(1, P, dtype=W.dtype, device=W.device)], dim=0)

        # Dehomogenize
        X = homog[0, :] / homog[3, :]
        Y = homog[1, :] / homog[3, :]
        Z = homog[2, :] / homog[3, :]

        # Centroid
        mean_x, mean_y, mean_z = X.mean(), Y.mean(), Z.mean()

        # Shift
        Xs = X - mean_x
        Ys = Y - mean_y
        Zs = Z - mean_z

        mean_dist = torch.mean(torch.sqrt(Xs**2 + Ys**2 + Zs**2))

        s = torch.sqrt(torch.tensor(3.0, dtype=W.dtype, device=W.device)) / mean_dist
        #print(3)

        T = torch.tensor([
            [s, 0, 0, -s*mean_x],
            [0, s, 0, -s*mean_y],
            [0, 0, s, -s*mean_z],
            [0, 0, 0, 1]
        ], dtype=W.dtype, device=W.device)

        homog_norm = T @ homog
        W_norm[3*f:3*f+3, :] = homog_norm[:3, :]

        T_list.append(T)

    return W_norm, T_list

# metric_upgrade_daq.py
import torch
from typing import List, Tuple

def _sym_vectorize_4x4(Q: torch.Tensor) -> torch.Tensor:
    """
    Symmetric 4x4 -> 10-vector [q11, q12, q13, q14, q22, q23, q24, q33, q34, q44].
    """
    return torch.stack([
        Q[0,0], Q[0,1], Q[0,2], Q[0,3],
                Q[1,1], Q[1,2], Q[1,3],
                        Q[2,2], Q[2,3],
                                Q[3,3]
    ])

def _place_sym_from_vec(q: torch.Tensor) -> torch.Tensor:
    """
    10-vector back to symmetric 4x4.
    """
    Q = torch.zeros((4,4), dtype=q.dtype, device=q.device)
    Q[0,0] = q[0]
    Q[0,1] = Q[1,0] = q[1]
    Q[0,2] = Q[2,0] = q[2]
    Q[0,3] = Q[3,0] = q[3]
    Q[1,1] = q[4]
    Q[1,2] = Q[2,1] = q[5]
    Q[1,3] = Q[3,1] = q[6]
    Q[2,2] = q[7]
    Q[2,3] = Q[3,2] = q[8]
    Q[3,3] = q[9]
    return Q

def _constraint_row_from_C(C: torch.Tensor, r: int, s: int) -> torch.Tensor:
    """
    Build the linear row a^T * vecs(Q) = 0 for the (r,s) entry of C Q C^T.
    Uses the 10-term symmetric parameterization of Q.
    """
    # Entry (r,s) of C Q C^T = sum_{j,k} C[r,j] * Q[j,k] * C[s,k].
    # For symmetric Q, collect the 10 unique terms with appropriate 2x for off-diagonals.
    a = torch.zeros(10, dtype=C.dtype, device=C.device)
    # helper to add contribution for (j,k)
    def add(j,k, coeff):
        idx_map = {
            (0,0): 0,
            (0,1): 1, (1,0): 1,
            (0,2): 2, (2,0): 2,
            (0,3): 3, (3,0): 3,
            (1,1): 4,
            (1,2): 5, (2,1): 5,
            (1,3): 6, (3,1): 6,
            (2,2): 7,
            (2,3): 8, (3,2): 8,
            (3,3): 9,
        }
        a[idx_map[(j,k)]] += coeff

    # accumulate symmetric terms
    for j in range(4):
        for k in range(4):
            coeff = C[r, j] * C[s, k]
            if j == k:
                add(j, k, coeff)               # diagonal once
            elif j < k:
                add(j, k, coeff)               # upper
            else:
                add(k, j, coeff)               # mirror into upper
    return a

def solve_dual_absolute_quadric(P: torch.Tensor, K_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Solve for Ω* (4x4 symmetric, rank 3) from constraints:
        C_i Ω* C_i^T ∝ I, where C_i = K_i^{-1} P_i
    Inputs:
      P: (3F, 4) stacked cameras (each block 3x4)
      K_list: list of F intrinsics (3x3)
    Returns:
      Omega_star: (4x4) dual absolute quadric, scaled so that its three non-zero eigenvalues are positive.
    """
    F = len(K_list)
    assert P.shape[0] == 3*F and P.shape[1] == 4
    device = P.device
    rows = []

    for f in range(F):
        Pf = P[3*f:3*f+3, :]                       # 3x4
        Kinv = torch.linalg.inv(K_list[f]).to(device)
        C = Kinv @ Pf                               # 3x4

        # W' = C Ω* C^T should be proportional to I:
        # Off-diagonals = 0  -> (0,1),(0,2),(1,2)
        rows.append(_constraint_row_from_C(C, 0, 1))
        rows.append(_constraint_row_from_C(C, 0, 2))
        rows.append(_constraint_row_from_C(C, 1, 2))
        # Equal diagonals -> W11 - W22 = 0, W11 - W33 = 0
        a11 = _constraint_row_from_C(C, 0, 0)
        a22 = _constraint_row_from_C(C, 1, 1)
        a33 = _constraint_row_from_C(C, 2, 2)
        rows.append(a11 - a22)
        rows.append(a11 - a33)

    A = torch.stack(rows, dim=0)                   # [(5F) x 10]

    # Solve A q = 0 (in least-squares sense): smallest singular vector
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    q = Vh[-1, :]                                  # 10-vector
    Omega_star = _place_sym_from_vec(q)

    # Project to rank-3 PSD with signature (+,+,+,0)
    evals, U = torch.linalg.eigh(Omega_star)
    # Sort ascending; last should be ~0, adjust numerically
    idx = torch.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]

    # Clamp negatives (numerical) on the top 3 to be positive, force the last to 0
    e = evals.clone()
    e[:-1] = torch.clamp(e[:-1], min=1e-12)
    e[-1] = 0.0
    Omega_star_psd = (U @ torch.diag(e) @ U.T)

    # Normalize (optional): scale so that mean of non-zero eigs is 1
    nz = e[:-1].mean().clamp(min=1e-12)
    Omega_star_psd = Omega_star_psd / nz

    return Omega_star_psd

def factor_upgrade_from_OmegaStar(Omega_star: torch.Tensor) -> torch.Tensor:
    """
    Find H such that Omega_star = H * diag(1,1,1,0) * H^T.
    Returns H (4x4).
    """
    evals, U = torch.linalg.eigh(Omega_star)
    idx = torch.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]
    # Build sqrt of eigenvalues for first three components
    s = torch.sqrt(torch.clamp(evals[:-1], min=1e-12))
    H = U @ torch.diag(torch.cat([s, torch.tensor([1.0], dtype=Omega_star.dtype, device=Omega_star.device)]))
    return H

def _project_to_SO3(M: torch.Tensor) -> torch.Tensor:
    """
    Nearest rotation to a 3x3 matrix via SVD (polar decomposition).
    """
    U, S, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, S.mean()

def metric_upgrade_daq(
    P: torch.Tensor,              # (3F,4) stacked projective cameras
    K_list: List[torch.Tensor],   # list of F intrinsics (3,3)
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute Euclidean upgrade H from projective cameras P, then return:
      H: (4,4)  such that P_e = P @ H^{-1} are Euclidean-form cameras
      P_e: (3F,4) upgraded cameras
      R_list: list of F rotations (3,3)
      t_list: list of F translations (3,)
    """
    device = P.device
    Omega_star = solve_dual_absolute_quadric(P, K_list)
    H = factor_upgrade_from_OmegaStar(Omega_star)
    Hinv = torch.linalg.inv(H)

    F = len(K_list)
    P_e = P @ Hinv                                     # upgraded cameras
    R_list, t_list = [], []

    for f in range(F):
        Pf = P_e[3*f:3*f+3, :]                         # 3x4
        K = K_list[f].to(device)
        Kinv = torch.linalg.inv(K)
        A = Kinv @ Pf                                  # should be [R | t]
        R_approx = A[:, :3]
        R, scale = _project_to_SO3(R_approx)                  # nearest rotation
        t = A[:, 3] / scale
        # Reassign translation consistent with projected R
        # (optional) refine t via least squares: K[R|t] ~ Pf
        t_list.append(t)
        R_list.append(R)

    return H, P_e, R_list, t_list
