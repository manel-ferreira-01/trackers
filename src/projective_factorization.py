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


def projective_factorization(obs_mat_scaled):
    """
    Performs Projective Factorization on a scaled observation matrix (W * Lambda).
    
    Args:
        obs_mat_scaled: (3*num_frames, num_points) - [u*L, v*L, L]^T structure.
        
    Returns:
        M: Metric motion matrix (3F x 3)
        S: Metric shape matrix (3 x P)
        T: Translation component (3F x 1)
    """
    device = obs_mat_scaled.device
    dtype = obs_mat_scaled.dtype
    num_frames = obs_mat_scaled.shape[0] // 3

    # 1. Centering (Recovery of Translation)
    # The translation is the mean of each row (if shape is zero-centered)
    T_vec = obs_mat_scaled.mean(dim=1, keepdim=True) # (3F, 1)
    W_centered = obs_mat_scaled - T_vec

    # 2. SVD to find rank-3 subspace
    U, S_vals, Vh = torch.linalg.svd(W_centered, full_matrices=False)
    
    S_root = torch.sqrt(S_vals[0:3])
    M_hat = U[:, 0:3] * S_root       # (3F, 3)
    S_hat = S_root[:, None] * Vh[0:3, :] # (3, P)

    # 3. Metric Constraints (Enforcing Orthogonality)
    # For each frame, the 3x3 block must be a rotation matrix (M_hat @ Q @ M_hat.T = I)
    A_rows = []
    b_rows = []
    def constraint_torch(m1, m2):
        """
        Helper to create the 6-parameter vector for the symmetric matrix Q.
        m1.T @ Q @ m2 = b
        """
        return torch.tensor([
            m1[0]*m2[0], m1[0]*m2[1] + m1[1]*m2[0], m1[0]*m2[2] + m1[2]*m2[0],
            m1[1]*m2[1], m1[1]*m2[2] + m1[2]*m2[1],
            m1[2]*m2[2]
        ], device=m1.device, dtype=m1.dtype)


    for f in range(num_frames):
        # Rows for X, Y, and Z (Depth/Scale row)
        ix, iy, iz = 3*f, 3*f+1, 3*f+2
        mx, my, mz = M_hat[ix], M_hat[iy], M_hat[iz]

        # Unit Norm constraints: |mx|^2=1, |my|^2=1, |mz|^2=1
        A_rows.append(constraint_torch(mx, mx)); b_rows.append(1.0)
        A_rows.append(constraint_torch(my, my)); b_rows.append(1.0)
        A_rows.append(constraint_torch(mz, mz)); b_rows.append(1.0)

        # Orthogonality constraints: mx.my=0, mx.mz=0, my.mz=0
        A_rows.append(constraint_torch(mx, my)); b_rows.append(0.0)
        A_rows.append(constraint_torch(mx, mz)); b_rows.append(0.0)
        A_rows.append(constraint_torch(my, mz)); b_rows.append(0.0)

    A = torch.stack(A_rows)
    b = torch.tensor(b_rows, device=device, dtype=dtype).unsqueeze(1)

    # Solve for Q (Linear Least Squares: A @ q = b)
    # q is the 6 parameters of the symmetric matrix Q = L @ L.T
    q = torch.linalg.lstsq(A, b).solution.flatten()

    Q = torch.tensor([
        [q[0], q[1], q[2]],
        [q[1], q[3], q[4]],
        [q[2], q[4], q[5]]
    ], device=device, dtype=dtype)

    # 4. Extract Transformation L
    # Ensure Q is positive definite (Numerical stability)
    eigvals, eigvecs = torch.linalg.eigh(Q)
    eigvals = torch.clamp(eigvals, min=1e-6)
    L = eigvecs @ torch.diag(torch.sqrt(eigvals))

    # 5. Final Metric Reconstruction
    M = M_hat @ L
    S = torch.linalg.inv(L) @ S_hat

    R_1 = M[0:3, :3]
    det = torch.linalg.det(R_1)

    if det < 0:
        # We are in a left-handed (mirrored) system.
        # Flip exactly ONE column of L. This changes the handedness 
        # of the transformation without breaking the orthogonality.
        L[:, 0] *= -1
        
        # Re-calculate M and S with the corrected L
        M = M_hat @ L
        S = torch.linalg.inv(L) @ S_hat

    return M, S, T_vec

