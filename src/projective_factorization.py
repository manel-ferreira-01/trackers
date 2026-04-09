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

    rays3d = make_homogenous(rays2d)  # [3F, P]

    return rays3d, z

def make_homogenous(obs_mat):
    F2, P = obs_mat.shape
    F = F2 // 2
    device = obs_mat.device
    dtype = obs_mat.dtype

    obs_reshaped = obs_mat.view(F, 2, P)

    ones = torch.ones((F, 1, P), device=device, dtype=dtype)

    obs_combined = torch.cat([obs_reshaped, ones], dim=1)
    obs_homog = obs_combined.view(3 * F, P)
    return obs_homog

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
            m1[0]*m2[0],
            m1[0]*m2[1] + m1[1]*m2[0],
            m1[0]*m2[2] + m1[2]*m2[0],
            m1[1]*m2[1],
            m1[1]*m2[2] + m1[2]*m2[1],
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
    b = torch.tensor(b_rows, device=device, dtype=dtype).unsqueeze(1) # (6*num_frames, 1)

    mat = torch.kron(
        torch.eye(num_frames, dtype=dtype, device=device), 
        torch.tensor([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device).reshape(6,1)
    )

    A = torch.cat([A, -b], dim=1)

    # Solve for Q (Linear Least Squares: A @ q = b)
    # q is the 6 parameters of the symmetric matrix Q = L @ L.T
    _, _, Vh_full = torch.linalg.svd(A)
    l = Vh_full[-1] # This is our [q1...q6, a1^2...aF^2]

    l = l / l[6] 
    
    q_vec = l[:6]
    #alphas_sq = l[6:]

    #print("alphas_sq", alphas_sq)

    Q = torch.tensor([
        [q_vec[0], q_vec[1], q_vec[2]],
        [q_vec[1], q_vec[3], q_vec[4]],
        [q_vec[2], q_vec[4], q_vec[5]]], device=device, dtype=dtype)

    # 4. Extract Transformation L
    # Ensure Q is positive definite (Numerical stability)
    U, S_diag, Vh = torch.linalg.svd(Q)
    S_diag = torch.clamp(S_diag, min=1e-9)
    L = U @ torch.diag(torch.sqrt(S_diag))

    # 5. Final Metric Reconstruction
    M = M_hat @ L
    S = torch.linalg.inv(L) @ S_hat

    # procrustes on the motion matrices
    scales = []
    for f in range(num_frames):
        Mi = M[3*f : 3*(f+1), :]
        U_m, S_m, Vh_m = torch.linalg.svd(Mi)
        R = U_m @ Vh_m
        if torch.linalg.det(R) < 0:
            R *= -1
        M[3*f : 3*(f+1), :] = R
        scales.append(S_m.mean())

    return M , S, T_vec, torch.stack(scales) #/ torch.sqrt(alphas_sq)

def projective_factorization_fast(obs_mat_scaled):
    device = obs_mat_scaled.device
    dtype = obs_mat_scaled.dtype
    F = obs_mat_scaled.shape[0] // 3

    # 1. Center
    T_vec = obs_mat_scaled.mean(dim=1, keepdim=True)
    W_centered = obs_mat_scaled - T_vec

    # 2. SVD rank-3
    U, S_vals, Vh = torch.linalg.svd(W_centered, full_matrices=False)
    S_root = torch.sqrt(S_vals[:3])
    M_hat = (U[:, :3] * S_root).contiguous()    # line ~26
    S_hat = (S_root[:, None] * Vh[:3]).contiguous()

    # 3. Metric constraints — fully vectorized, no loop
    # Reshape M_hat into (F, 3, 3): rows are [mx, my, mz] per frame
    Mf = M_hat.reshape(F, 3, 3)           # (F, 3, 3)

    # Build all outer products at once: (F, 3, 3, 3) -> constraint vectors (F, 3, 3, 6)
    # For each pair (i,j), constraint vector c_ij has 6 entries for symmetric Q
    # c = [m[a]*m[b]] mapped to upper-tri indices
    # Indices for symmetric 3x3: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    i_idx = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
    j_idx = torch.tensor([0, 1, 2, 1, 2, 2], device=device)

    # m_i, m_j: (F, 3, 3) -> pick components
    mi = Mf[:, :, i_idx]   # (F, 3, 6)  — axis1=which row (mx/my/mz), axis2=component index
    mj = Mf[:, :, j_idx]   # (F, 3, 6)

    # Symmetry factor: off-diagonal gets factor 2 (m[a]*m[b] + m[b]*m[a])
    sym_factor = torch.where(i_idx == j_idx,
                             torch.ones(6, device=device, dtype=dtype),
                             2 * torch.ones(6, device=device, dtype=dtype))

    # A_constraints: (F, 9, 6) — 9 constraints per frame (3 norm + 6 ortho... actually 3+3+3=9)
    # Each row pair (row_a, row_b): sum over components with sym_factor
    A_block = (mi * mj * sym_factor).sum(dim=0)  # wrong — need per-pair

    # Correct: for each frame, constraint for row pair (a,b) is sum_k m[a,k]*m[b,k] for each Q entry
    # A[f, constraint_idx, q_idx] = sum_k  mi[f,a,k] * mj[f,b,k]  -- but we want Q contraction
    # Let's do it directly:
    # For pair (a,b): c_vec[q] = m_a[i_idx[q]] * m_b[j_idx[q]] * (1 if diag else 1, sum both)
    # Since Q is symmetric: m^T Q m = sum_{p,q} m[p] Q[p,q] m[q]
    #                                = sum over upper-tri: q_vec[k] * (m[i_k]*m[j_k] * sym[k])

    pairs = [(0,0,1.0),(1,1,1.0),(2,2,1.0),(0,1,0.0),(0,2,0.0),(1,2,0.0)]
    b_vals = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    # m_a, m_b for all pairs at once: pair_a/b indices
    pa = torch.tensor([p[0] for p in pairs], device=device)
    pb = torch.tensor([p[1] for p in pairs], device=device)
    b_vec = torch.tensor(b_vals, device=device, dtype=dtype)

    ma_all = Mf[:, pa, :]   # (F, 6, 3)
    mb_all = Mf[:, pb, :]   # (F, 6, 3)

    # A_block[f, constraint, q_entry] = ma[f,c,i_idx[q]] * mb[f,c,j_idx[q]] * sym[q]
    A_constraints = (ma_all[:, :, i_idx] * mb_all[:, :, j_idx]) * sym_factor  # (F, 6, 6)
    A_constraints = A_constraints.view(F * 6, 6)

    b_full = b_vec.unsqueeze(0).expand(F, -1).reshape(F * 6, 1)

    # Append homogeneous column and solve null space
    A_aug = torch.cat([A_constraints, -b_full], dim=1)   # (6F, 7)

    _, _, Vh_full = torch.linalg.svd(A_aug)
    l = Vh_full[-1]
    l = l / l[6]
    q_vec = l[:6]

    Q = torch.zeros(3, 3, device=device, dtype=dtype)
    Q[i_idx, j_idx] = q_vec
    Q[j_idx, i_idx] = q_vec   # symmetrize

    # 4. Factor Q -> L
    Uq, Sq, _ = torch.linalg.svd(Q)
    Sq = torch.clamp(Sq, min=1e-9)
    L = Uq @ torch.diag(torch.sqrt(Sq))

    M = M_hat @ L
    S = torch.linalg.solve(L, S_hat)   # faster than inv @ S_hat

    # 5. Batched Procrustes — no loop
    Mf2 = M.reshape(F, 3, 3)                             # (F, 3, 3)
    Up, Sp, Vhp = torch.linalg.svd(Mf2)               # all F at once
    R_batch = Up @ Vhp                                 # (F, 3, 3)
    #R_batch = Mf2

    # Fix det < 0
    #dets = torch.linalg.det(R_batch)                   # (F,)
    #flip = torch.ones(F, 1, 1, device=device, dtype=dtype)
    #flip[dets < 0] = 1.0
    #R_batch = R_batch * flip

    M =R_batch.reshape(3 * F, 3)
    scales = Sp.mean(dim=1)                            # (F,)

    return  M, S, T_vec, Sp

import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_subspace_outlier_indices(W, rank=4, threshold=100.0, use_relative=False, 
                                 mode='svd', iterations=100,ransac_threshold=1, viz=True):
    """
    Detects column outliers by projecting data onto a subspace.
    
    Modes:
        'svd'   : Standard SVD-based projection (sensitive to outliers).
        'ransac': Robustly samples columns to find the best subspace.
    """
    device = W.device
    D, N = W.shape
    
    if mode == 'svd':
        # 1. Standard SVD to find the basis
        u, s, v = torch.linalg.svd(W, full_matrices=False)
        Ub = u[:, :rank]
        
        # 2. Project onto null space and calculate residuals
        p_null = torch.eye(D, device=device) - Ub @ Ub.T
        residuals = torch.norm(p_null @ W, dim=0)**2 # Squared Frobenius norm
        
    elif mode == 'ransac':
        best_inlier_count = -1
        best_mask = None
        sample_size = rank # Minimum columns to define a rank-r subspace
        
        for i in range(iterations):
            # 1. Randomly sample columns
            perm = torch.randperm(N, device=device)[:sample_size]
            W_sample = W[:, perm]
            
            # 2. Get basis via QR (faster than SVD for small tall matrices)
            # Q is the orthogonal basis for the sampled columns
            Q, _ = torch.linalg.qr(W_sample)
            Ub_h = Q[:, :rank]
            
            # 3. Project ALL columns onto null space of this hypothesis
            p_null_h = torch.eye(D, device=device) - Ub_h @ Ub_h.T
            res_h = torch.norm(p_null_h @ W, dim=0)**2
            
            # 4. Determine inliers based on current threshold
            # Note: In RANSAC, 'threshold' usually refers to the distance error
            inlier_mask = res_h < ransac_threshold
            num_inliers = inlier_mask.sum()
            
            if num_inliers > best_inlier_count:
                best_inlier_count = num_inliers
                best_mask = inlier_mask
        
        # 5. Refinement: Re-estimate basis using ONLY the best inliers
        W_inliers = W[:, best_mask]
        u_final, _, _ = torch.linalg.svd(W_inliers, full_matrices=False)
        Ub_final = u_final[:, :rank]
        
        p_null = torch.eye(D, device=device) - Ub_final @ Ub_final.T
        residuals = torch.norm(p_null @ W, dim=0)**2
        #print(f"RANSAC finished {iterations} iterations. Inliers found: {best_inlier_count}")

    # --- Thresholding Logic ---
    if use_relative:
        cutoff = torch.median(residuals) * threshold 
    else:
        cutoff = threshold
        
    outlier_mask = residuals > cutoff
    outlier_indices = torch.nonzero(outlier_mask).flatten()
    new_W = W[:, ~outlier_mask]
    
    if viz:
        plt.figure(figsize=(10, 4))
        plt.plot(residuals.cpu().numpy(), label="Residuals", alpha=0.7)
        plt.axhline(cutoff.cpu().item() if torch.is_tensor(cutoff) else cutoff, 
                    color='red', linestyle='--', label=f"Threshold ({mode})")
        plt.yscale("log")
        plt.title(f"Outlier Detection ({mode.upper()})")
        plt.xlabel("Column Index")
        plt.legend()
        plt.show()
        print(f"Number of outliers detected: {outlier_mask.sum().item()}")
        
    return new_W, outlier_mask, residuals

def compare_3x4_trajectories(cam_lists, gt_lists, min_t_mag=0.01):
    
    def to_4x4(m):
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        return np.vstack([m, [0, 0, 0, 1]])

    def normalize(traj):
        m4x4 = [to_4x4(m) for m in traj]
        m0_inv = np.linalg.inv(m4x4[0])
        return [m @ m0_inv for m in m4x4]

    rel_alg = normalize(cam_lists)
    rel_gt  = normalize(gt_lists)

    #print("rel_alg",torch.tensor(rel_alg))
    #print("rel_gt", torch.tensor(rel_gt))

    #print(torch.tensor(rel_alg) - torch.tensor(rel_gt))

    print([np.linalg.det(cam[:3,:3]) for cam in rel_alg])


    rot_errors  = []
    dir_errors  = []
    norms_alg   = []  # ← collect here
    norms_gt    = []  # ← collect here

    for i in range(1, len(rel_alg)):
        # --- Rotation Error ---
        R_alg = rel_alg[i][:3, :3]
        R_gt  = rel_gt[i][:3, :3]

        #print(np.linalg.det(R_alg), np.linalg.det(R_gt))

        R_diff    = R_alg @ R_gt.T
        trace_val = (np.trace(R_diff) - 1) / 2.0
        print(trace_val)
        rot_err   = np.degrees(np.arccos((trace_val)))
        rot_errors.append(rot_err)

        # --- Translation ---
        t_alg    = rel_alg[i][:3, 3]
        t_gt     = rel_gt[i][:3, 3]
        norm_alg = np.linalg.norm(t_alg)
        norm_gt  = np.linalg.norm(t_gt)

        norms_alg.append(norm_alg)  # ← save
        norms_gt.append(norm_gt)    # ← save

        if norm_alg < min_t_mag or norm_gt < min_t_mag:
            dir_errors.append(np.nan)   # exclude near-zero baselines
        else:
            unit_alg = t_alg / norm_alg
            unit_gt  = t_gt  / norm_gt
            dot      = np.dot(unit_alg, unit_gt)
            dir_errors.append(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))

    return {
        "mean_rot":  np.mean(rot_errors),
        "mean_dir":  np.nanmean(dir_errors),
        "rot_list":  rot_errors,
        "dir_list":  dir_errors,
        "norm_alg":  norms_alg,   # (F-1,) one per relative frame
        "norm_gt":   norms_gt,
    }