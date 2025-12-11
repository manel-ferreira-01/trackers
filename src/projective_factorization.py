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


def build_depth_weighted_matrix(tracks, depths: torch.Tensor, Ks):
    """
    Build a 3F x P depth-weighted observation matrix with per-frame intrinsics.

    Args:
        tracks: torch.Tensor [2F, P]
            Pixel coordinates (u,v) for each feature across frames.
            Rows 0,2,4... are u; Rows 1,3,5... are v.
        depths: torch.Tensor [F, H, W]
            Depth map for each frame.
        Ks: torch.Tensor [F, 3, 3]
            Camera intrinsics matrix for each frame.

    Returns:
        W_proj: torch.Tensor [3F, P]
            Depth-weighted homogeneous observation matrix.
        z: torch.Tensor [F, P]
            Sampled depth values.
        T: torch.Tensor [F, 3, 3]
            Hartley normalization transforms for each frame.
    """
    F = depths.shape[0]
    P = tracks.shape[1]

    # repeat the K if only one is given
    #if Ks.ndim == 2: 
    #    Ks = Ks.unsqueeze(0).expand(F, -1, -1)
    # ----------------------
    
    # 1. Sample depths
    z = sample_depths(depths, tracks) 

    # 2. Extract u, v coordinates
    u = tracks[0::2, :]
    v = tracks[1::2, :]

    # 3. Extract intrinsics per frame (Ks is now guaranteed to be [F, 3, 3])
    fx = Ks[:, 0, 0].unsqueeze(1) 
    fy = Ks[:, 1, 1].unsqueeze(1)
    cx = Ks[:, 0, 2].unsqueeze(1)
    cy = Ks[:, 1, 2].unsqueeze(1)

    # ... rest of the function remains the same ...
    x_norm = (u - cx) / fx 
    y_norm = (v - cy) / fy 

    rays2d = torch.stack([x_norm, y_norm], dim=1)
    rays2d = rays2d.reshape(2 * F, P)

    rays2d_norm, T = hartley_normalize_stacked(rays2d)

    W_proj = rays2d_norm * z.repeat_interleave(2, dim=0)

    return W_proj, z, T, rays2d


def hartley_normalize_stacked(W):
    """
    Applies Hartley's isotropic normalization to a stacked measurement matrix.
    
    Args:
        W: Tensor of shape (2*F, P). 
           Rows 2*f are x-coords, Rows 2*f+1 are y-coords.
           
    Returns:
        W_norm: Tensor (2*F, P) centered and scaled.
        T: Tensor (F, 3, 3) the transformation matrices used (for inversion).
    """
    TwoF, P = W.shape
    F = TwoF // 2
    device = W.device

    # 1. Reshape to (F, 2, P) to separate X and Y easily
    # This view allows us to calculate means across the P dimension
    W_view = W.view(F, 2, P)
    
    # 2. Compute Centroid per frame
    # Mean across points (dim 2). Shape: (F, 2, 1)
    centroid = torch.mean(W_view, dim=2, keepdim=True)
    
    # 3. Shift to Origin
    centered = W_view - centroid

    # 4. Compute Scale
    # Distance of every point from origin: sqrt(x^2 + y^2)
    # Shape: (F, P)
    dist = torch.norm(centered, dim=1) 
    
    # Mean distance per frame. Shape: (F, 1)
    mean_dist = torch.mean(dist, dim=1, keepdim=True)
    
    # Scale factor
    scale = 1.41421356 / mean_dist
    
    # 5. Apply Scale
    # Shape: (F, 2, P)
    W_norm_view = centered * scale.unsqueeze(2)
    
    # 6. Flatten back to (2F, P)
    W_norm = W_norm_view.reshape(TwoF, P)

    # 7. Construct Transformation Matrix T (for inversion later)
    # T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    T = torch.eye(3, device=device).repeat(F, 1, 1)
    
    s = scale.squeeze()             # (F,)
    cx = centroid[:, 0, 0]          # (F,)
    cy = centroid[:, 1, 0]          # (F,)

    T[:, 0, 0] = s
    T[:, 1, 1] = s
    T[:, 0, 2] = -s * cx
    T[:, 1, 2] = -s * cy
    
    return W_norm, T

import numpy as np

import numpy as np

def solve_metric_upgrade_rays(P_hat_stack):
    """
    Computes H for metric upgrade with a fix for SVD sign ambiguity.
    """
    m = P_hat_stack.shape[0]
    A = np.zeros((5 * m, 10))

    # Index mapping for symmetric matrix Q
    idx_map = {
        (0,0):0, (0,1):1, (0,2):2, (0,3):3,
        (1,1):4, (1,2):5, (1,3):6,
        (2,2):7, (2,3):8,
        (3,3):9
    }

    def get_coeffs(u, v):
        coeffs = np.zeros(10)
        for r in range(4):
            for c in range(r, 4):
                k = idx_map[(r, c)]
                if r == c:
                    coeffs[k] = u[r] * v[r]
                else:
                    coeffs[k] = u[r] * v[c] + u[c] * v[r]
        return coeffs

    # Build A
    row_idx = 0
    for i in range(m):
        P = P_hat_stack[i]
        r1, r2, r3 = P[0], P[1], P[2]
        # R1 is 1x4

        # 1. Orthogonality
        A[row_idx] =    (r1, r2); row_idx += 1
        A[row_idx] = get_coeffs(r1, r3); row_idx += 1
        A[row_idx] = get_coeffs(r2, r3); row_idx += 1
        # 2. Aspect Ratio
        A[row_idx] = get_coeffs(r1, r1) - get_coeffs(r2, r2); row_idx += 1
        A[row_idx] = get_coeffs(r2, r2) - get_coeffs(r3, r3); row_idx += 1

    # Solve Ax=0
    _, _, Vt = np.linalg.svd(A)
    x = Vt[-1]

    # --- Reconstruction of Q ---
    def make_Q(x_vec):
        Q_mat = np.zeros((4, 4))
        Q_mat[0,0], Q_mat[1,1], Q_mat[2,2], Q_mat[3,3] = x_vec[0], x_vec[4], x_vec[7], x_vec[9]
        Q_mat[0,1] = Q_mat[1,0] = x_vec[1]
        Q_mat[0,2] = Q_mat[2,0] = x_vec[2]
        Q_mat[0,3] = Q_mat[3,0] = x_vec[3]
        Q_mat[1,2] = Q_mat[2,1] = x_vec[5]
        Q_mat[1,3] = Q_mat[3,1] = x_vec[6]
        Q_mat[2,3] = Q_mat[3,2] = x_vec[8]
        return Q_mat

    Q = make_Q(x)

    # --- FIX START: Detect Sign Flip ---
    # Q must be Positive Semi-Definite. Check the eigenvalues.
    evals_raw, _ = np.linalg.eigh(Q)
    
    # If the largest magnitude eigenvalue is negative, the whole matrix is flipped.
    # We check the one with largest absolute value to decide the dominant sign.
    if evals_raw[np.argmax(np.abs(evals_raw))] < 0:
        x = -x
        Q = make_Q(x)
    # --- FIX END ---

    # Decompose Q (Symmetric)
    evals, evecs = np.linalg.eigh(Q)
    
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Truncate to Rank 3
    # Note: If data is noisy, 4th eigenvalue might be slightly negative, 
    # so we clip to 0. But top 3 MUST be positive now.
    S_sqrt = np.diag(np.sqrt(np.maximum(evals[:3], 1e-10)))
    H3 = evecs[:, :3] @ S_sqrt

    # 4th vector (Null space) for h
    h = evecs[:, 3:4]
    
    H = np.hstack([H3, h])
    return H