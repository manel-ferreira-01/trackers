import torch
import numpy as np
from tqdm import tqdm
from src.projective_factorization import projective_factorization


def marques_factorization(obs_mat: torch.Tensor):
    """
    Performs Marques Factorization on an interleaved observation matrix.
    
    Args:
        obs_mat: torch.Tensor of shape (2*num_frame, num_features)
                 Structure is interleaved: [x0, y0, x1, y1, ...]^T

    Returns:
        M: motion matrix (2*num_frames x 3) - Interleaved
        S: shape matrix (3 x num_features)
        tvecs: translation vectors per frame (num_frames x 2)
        alphas: scaling parameters per frame (num_frames,)
    """

    if type(obs_mat) is np.ndarray:
        obs_mat = torch.tensor(obs_mat, dtype=torch.float32)
    device = obs_mat.device
    dtype = obs_mat.dtype
    
    # 1. Calculate centroids (Translation vectors)
    # Even rows are Xs, Odd rows are Ys
    Xs = obs_mat[0::2, :]
    Ys = obs_mat[1::2, :]
    
    # tvecs shape: (num_frames, 2) -> [[tx0, ty0], [tx1, ty1], ...]
    tvecs = torch.stack([Xs.mean(dim=1), Ys.mean(dim=1)], dim=1) 
    
    # 2. Center observations (Maintain Interleaved Structure)
    # We clone to avoid modifying the input in place if that matters, 
    # or we can modify obs_mat directly if memory is tight.
    obs_mat_centered = obs_mat.clone()
    
    # Subtract Tx from even rows
    obs_mat_centered[0::2, :] = Xs - tvecs[:, 0:1]
    # Subtract Ty from odd rows
    obs_mat_centered[1::2, :] = Ys - tvecs[:, 1:2]

    # 3. SVD decomposition on Interleaved Matrix
    # U will be (2F, 2F), we take top 3 components
    U, S, Vh = torch.linalg.svd(obs_mat_centered, full_matrices=True)
    V = Vh.T

    S_root = torch.sqrt(S[0:3])
    # M_hat is now interleaved: [Mx0, My0, Mx1, My1, ...]
    M_hat = U[:, 0:3] * S_root
    S_hat = S_root[:, None] * V[:, 0:3].T

    num_frames = obs_mat.shape[0] // 2
    A_rows = []
    b = []

    # 4. Build Constraints (Interleaved Indexing)
    for f in range(num_frames):
        # Row indices for frame f in an interleaved matrix
        row_x = 2 * f
        row_y = 2 * f + 1
        
        # Constraint: |Mx|^2 = alpha
        A_rows.append(constraint_torch(M_hat[row_x], M_hat[row_x]))
        b.append(1.0)
        
        # Constraint: |My|^2 = alpha
        A_rows.append(constraint_torch(M_hat[row_y], M_hat[row_y]))
        b.append(1.0)
        
        # Constraint: Mx . My = 0
        A_rows.append(constraint_torch(M_hat[row_x], M_hat[row_y]))
        b.append(0.0)

    A = torch.stack(A_rows, dim=0)  # (3*num_frames, 6)
    b = torch.tensor(b, dtype=dtype, device=device).unsqueeze(1)  # (3*num_frames, 1)

    # Build the alpha constraints (Metr3ic constraints)
    # This structure remains valid: for every frame we have 3 equations, 
    # and we want the first two (norms) to be equal to alpha.
    mat = torch.kron(
        torch.eye(num_frames, dtype=dtype, device=device), 
        torch.tensor([[-1.0, -1.0, 0.0]], dtype=dtype, device=device).reshape(3,1)
    )
    A = torch.cat([A, mat], dim=1)

    # Solve homogeneous system A x = 0 via SVD
    _, _, Vh_full = torch.linalg.svd(A)
    l = Vh_full[-1]
    
    # Normalize and extract alphas
    # Check for division by zero in robust implementations, but keeping strictly to logic here
    l = l / l[6]  
    alphas = l[6:]

    # Build Q matrix and compute its Cholesky factor L
    Q = torch.tensor([
        [l[0], l[1], l[2]],
        [l[1], l[3], l[4]],
        [l[2], l[4], l[5]]
    ], dtype=dtype, device=device)

    if torch.trace(Q) < 0:
        Q = -Q

    # --- FIX START: Robust Cholesky / Square Root ---
    try:
        # 1. Try standard Cholesky first (fastest)
        L = torch.linalg.cholesky(Q)
    except RuntimeError:
        # 2. If it fails, force Positive Definiteness via Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(Q)
        
        # Clamp negative eigenvalues to a small positive epsilon
        epsilon = 1e-4
        eigvals = torch.where(eigvals > epsilon, eigvals, torch.tensor(epsilon, device=device, dtype=dtype))
        
        # Reconstruct L: Q = V * S * V.T  =>  L = V * sqrt(S)
        # Note: We don't need to reconstruct Q, we just need L such that L @ L.T approx Q
        L = eigvecs @ torch.diag(torch.sqrt(eigvals))

    # Recover Metric Motion and Shape
    M = M_hat @ L
    S = torch.linalg.inv(L) @ S_hat
    
    num_features = S.shape[1]

    # 5. Reconstruct Translation Matrix (Interleaved)
    # tvecs is (F, 2). view(-1, 1) makes it (2F, 1) -> [tx0, ty0, tx1, ty1...]
    T_vec = tvecs.view(-1, 1)
    T_mat = T_vec.repeat(1, num_features)

    # Reconstruct the full observation matrix W (Interleaved)
    W = (M @ S) + T_mat

    return W, M, S, tvecs, alphas


def constraint_torch(m1, m2):
    return torch.tensor([
        m1[0] * m2[0],
        m1[0] * m2[1] + m1[1] * m2[0],
        m1[0] * m2[2] + m1[2] * m2[0],
        m1[1] * m2[1],
        m1[1] * m2[2] + m1[2] * m2[1],
        m1[2] * m2[2]
    ], dtype=m1.dtype, device=m1.device)


def costeira_marques(Wo_np, iterMax1=50, iterMax2=30, stopError1=1e-5, stopError2=1e-2, device='cpu',
                     verbose=False):
    """
    Marques & Costeira factorization method (CV IU 2009) in PyTorch

    Args:
        Wo_np: np.ndarray or torch.Tensor with shape (2F, P), containing NaNs for missing entries
    Returns:
        Motion: (2F, 3) motion matrix
        Shape: (3, P) 3D shape matrix
        T: (2F, 1) translation per frame
    """
    if isinstance(Wo_np, np.ndarray):
        Wo = torch.tensor(Wo_np, dtype=torch.float32)
    else:
        Wo = Wo_np.clone().float()

    Wo = Wo.to(device)
    M = ~torch.isnan(Wo)  # mask of observed entries
    Wo[~M] = 0  # zero-fill missing entries for initialization

    row_means = torch.sum(Wo, dim=1) / torch.sum(M, dim=1)
    Wo = M * Wo + (~M) * row_means[:, None]  # mean imputation
    W = Wo.clone()

    F = W.shape[0] // 2 # Number of frames
    P = W.shape[1]

    ind = torch.where(torch.sum(M, dim=0) == 2 * F)[0]
    if len(ind) > 0:
        T = Wo[:, ind[0]].unsqueeze(1)
    else:
        T = W.mean(dim=1, keepdim=True)

    # Initial SVD factorization
    W_centered = W - T
    U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
    E = torch.diag(S[:3])
    R = U[:, :3] @ torch.sqrt(E)
    S = torch.sqrt(E) @ Vh[:3, :]

    iter1 = 0
    error1 = float('inf')
    Motionret = Tret = Shaperet = None

    try:
        with tqdm(total=iterMax1, desc='Outer Iterations') as pbar:
            while error1 > stopError1 and iter1 < iterMax1:
                W_centered = W - T
                Woit = Wo - T

                iterAux = 0
                error2 = float('inf')

                while error2 > stopError2 and iterAux < iterMax2:
                    Motion = []
                    for i in range(F):
                        A_f = proj_stiefel(R[2 * i:2 * i + 2, :].T).T
                        Motion.append(A_f)
                    Motion = torch.cat(Motion, dim=0)

                    Shape = torch.linalg.pinv(Motion) @ W_centered

                    prev_R = R.clone()
                    R = W_centered @ torch.linalg.pinv(Shape)

                    error2 = torch.norm(R - prev_R) / torch.norm(prev_R)
                    if verbose:
                        print(f"Inner Iteration {iterAux}, Error: {error2.item()}")

                    iterAux += 1

                W_hat = Motion @ Shape + T
                W = Motion @ Shape * (~M) + Woit * M + T

                iter1 += 1
                pbar.update(1)
                prev_error = error1
                error1 = torch.norm(W - W_hat) / torch.sqrt(torch.tensor(W.numel(), dtype=torch.float32))
                if verbose:
                    print(f"Outer Iteration {iter1}, Error: {error1.item()}")

                if len(ind) > 0:
                    T = Wo[:, ind[0]].unsqueeze(1)
                else:
                    T = W.mean(dim=1, keepdim=True)
                    Motionret = Motion
                    Tret = T
                    Shaperet = Shape

    except Exception as e:
        print("Fallback due to error:", e)
        Motion = Motionret
        T = Tret
        Shape = Shaperet

    return Motion, Shape, T, W

def proj_stiefel(Wo):
    U, S, Vh = torch.linalg.svd(Wo, full_matrices=False)
    c = S.mean()
    return c * U @ Vh

def _norm_uv_per_frame(tracks, K):
    """
    Normalize pixel coordinates per frame using either a single K or per-frame Ks.

    Args:
        tracks : [2F, P]
        K      : [3,3] or [F,3,3]

    Returns:
        x, y : [F, P] normalized coordinates
    """
    F = tracks.shape[0] // 2
    u = tracks[0::2, :]  # [F, P]
    v = tracks[1::2, :]  # [F, P]

    if K.ndim == 2:  # same intrinsics for all frames
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        #x = (u - cx) / fx
        #y = (v - cy) / fy
        x = u
        y = v
    else:
        # per-frame Ks
        fx = K[:, 0, 0].unsqueeze(1)  # [F,1]
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        #x = (u - cx) / fx
        #y = (v - cy) / fy
        x = u
        y = v

    return x, y

@torch.no_grad()
def _update_affine_ortho(x, y, lam, M, eps=1e-6, clamp_pos=True):
    F, P = lam.shape
    Mx, My = M[0::3], M[1::3]

    X2Y2 = x**2 + y**2              # [F,P]
    L = lam

    a11 = (L*L * X2Y2).sum(1) + eps
    a22 = (X2Y2).sum(1) + eps
    a12 = (L * X2Y2).sum(1)

    b1 = ((L*x)*Mx + (L*y)*My).sum(1)
    b2 = (x*Mx + y*My).sum(1)

    det = a11*a22 - a12*a12 + eps
    d = (a22*b1 - a12*b2) / det
    s = (-a12*b1 + a11*b2) / det
    if clamp_pos:
        d = torch.clamp(d, min=1e-5)
    return d, s

def _update_affine_ortho_lstsq(x, y, lam, M):
    """
    Orthographic per-frame least squares update using the full matrix solver.

    Args:
        x, y : [F, P] normalized image coordinates
        lam  : [F, P] raw monocular depths (λ)
        M    : [2F, P] target low-rank reconstruction (USV^T)

    Returns:
        d, s : [F] scale and shift per frame
    """
    F, P = lam.shape
    Mx, My, Mz = M[0::3], M[1::3], M[2::3]   # split 2×P blocks

    d = torch.empty(F, device=lam.device, dtype=lam.dtype)
    s = torch.empty(F, device=lam.device, dtype=lam.dtype)

    for f in range(F):
        # Design matrix A_f : [2P, 2]
        A_f = torch.stack([
            torch.cat([x[f] * lam[f], y[f] * lam[f], lam[f]]),   # column 1
            torch.cat([x[f], y[f], torch.ones_like(lam[f])])                      # column 2
        ], dim=1).reshape(3*P, 2)
        # Target vector B_f : [2P]
        B_f = torch.cat([Mx[f], My[f], Mz[f]])

        # Solve least squares (min ||A_f θ - B_f||^2)
        theta_f, *_ = torch.linalg.lstsq(A_f, B_f)
        d[f], s[f] = theta_f

    return d, s

def _update_projective_shift_only(tracks_homog, lam, M):
    """
    Projective per-frame least squares update for shift 's'.
    Works with 3 lines per frame: [x, y, 1].
    
    Args:
        tracks_homog : [3F, P] matrix [x0, y0, 1, x1, y1, 1, ...]^T
        lam          : [F, P] depths
        M            : [3F, P] low-rank reconstruction
        
    Returns:
        s : [F] shift per frame
    """
    F, P = lam.shape
    device = lam.device
    dtype = lam.dtype
    
    s = torch.empty(F, device=device, dtype=dtype)

    for f in range(F):
        # 1. Extract the 3 rows for this frame
        # tracks_homog is [x, y, 1] interleaved
        u_frame = tracks_homog[3*f : 3*f+3, :] # [3, P]
        m_frame = M[3*f : 3*f+3, :]            # [3, P]
        
        # 2. Flatten to vectors for dot product
        # A is the regressor: [x, y, 1]
        A = u_frame.reshape(-1) 
        
        # 3. Target B is (LowRank M) - (lambda * [x, y, 1])
        target_val = m_frame.reshape(-1)
        depth_component = (lam[f].repeat(3) * A) # lambda multiplied across x, y, 1
        
        # Correctly applying lambda: each lambda[f,p] scales [x_fp, y_fp, 1]
        # Using broadcasting for clarity:
        rhs_frame = m_frame - (lam[f] * u_frame)
        B = rhs_frame.reshape(-1)

        # 4. Solve: s = (A . B) / (A . A)
        numerator = torch.dot(A, B)
        denominator = torch.dot(A, A)
        
        s[f] = numerator / (denominator + 1e-8)

    return s


def calibrate_orthographic(tracks, lam, K, rank=4, iters=10, tol=1e-5, ridge=1e-6,
                           init_scales=None, init_offsets=None):

    #x, y = _norm_uv_per_frame(tracks, K)
    x , y = tracks[0::3,:], tracks[1::3,:] 

    F, P = lam.shape

    if init_scales is not None:
        d = init_scales.clone()
    else:
        d = torch.ones(F, device=lam.device, dtype=lam.dtype)  # scale
    if init_offsets is not None:
        s = init_offsets.clone()
    else:
        s = torch.zeros(F, device=lam.device, dtype=lam.dtype)  #offset

    scales = []
    offsets = []

    scales.append(d.clone())
    offsets.append(s.clone())

    best = (float('inf'), d.clone(), s.clone(), None)
    Mprev = None
    
    first_iter_W = (lam.repeat_interleave(3, dim=0) + offsets[-1][:,None].repeat_interleave(3,0) @ torch.ones(1, P)) * scales[-1][:,None].repeat_interleave(3,0) * tracks
    
    print(first_iter_W.shape)

    for iter in tqdm(range(iters)):

        W= (lam.repeat_interleave(3, dim=0) + offsets[-1][:,None].repeat_interleave(3,0) @ torch.ones(1, P)) * scales[-1][:,None].repeat_interleave(3,0) * tracks
        
        if 0:
            motion, shape, tvec, _ = projective_factorization(Wn)
            Wn = torch.cat((motion, tvec), dim=1) @ torch.cat((shape, torch.ones(1, P)), dim=0)
        else:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            M = (U[:, :rank] * S[:rank]) @ Vh[:rank]


        _, s = _update_affine_ortho_lstsq(x, y, lam, M)
        #s = _update_projective_shift_only(tracks,lam,M)
        #_, s = _update_affine_ortho(x, y, lam, M)

        # normalize d and s to avoid numerical issues, 
        #d = d / torch.norm(d)
        d = torch.ones_like(d)
        #s = s - s[0]

        scales.append(d.clone())
        offsets.append(s.clone())

        Wn = (lam.repeat_interleave(3, dim=0) + offsets[-1][:,None].repeat_interleave(3,0) @ torch.ones(1, P)) * scales[-1][:,None].repeat_interleave(3,0) * tracks
        
        #rho = (torch.norm(Wn - M) / (torch.norm(Wn) + 1e-12)).item()
        rho = (torch.norm(Wn - M)).item()
        if rho < best[0] - tol:
            best = (rho, d.clone(), s.clone(), M.clone())
        else:
            if iter > 10000:
                print(rho)
                break

    _, d, s, M = best
    W = (lam.repeat_interleave(3, dim=0) + offsets[-1][:,None].repeat_interleave(3,0) @ torch.ones(1, P)) * scales[-1][:,None].repeat_interleave(3,0) * tracks
    scales = torch.stack(scales)
    offsets = torch.stack(offsets)
    return scales, offsets, W, first_iter_W

import torch

def random_camera(camera_type="affine", device=None, dtype=torch.float32):
    """
    Returns a random camera matrix.
    
    Args:
        camera_type (str): "affine" (2x4) or "projective" (3x4).
        device: torch device.
        dtype: torch data type.
        
    Returns:
        P: The camera matrix.
    """
    if device is None:
        device = torch.device("cpu")

    # --- 1. Generate Rotation (Common to both) ---
    # Using QR decomposition of a random matrix to get a valid rotation matrix R
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    #d = torch.sign(torch.diag(R))
    #d[d == 0] = 1.0
    R_ext = Q #@ torch.diag(d)

    # --- 2. Generate Translation ---
    t = torch.randn(3, 1, device=device, dtype=dtype)

    if camera_type == "affine":
        # Affine P = [M | T] where M is 2x3 and T is 2x1
        # We take the first two rows of the rotation and translation
        M_f = R_ext[:2, :] 
        T_f = t[:2, :]
        P = torch.cat((M_f, T_f), dim=1) # Shape: [2, 4]

    elif camera_type == "projective":
        # Projective P = K [R | t]
        # Generate a random Intrinsic Matrix K (Upper triangular)
        # Standard defaults: focal length ~500, principal point ~250
        f = torch.rand(1, device=device, dtype=dtype) * 500 + 250
        px, py = torch.rand(2, device=device, dtype=dtype) * 100 + 250
        
        K = torch.tensor([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], device=device, dtype=dtype)
        
        # Extrinsic matrix [R | t]
        Rt = torch.cat((R_ext, t), dim=1) # Shape: [3, 4]
        
        # Final P = K @ Rt
        P = torch.eye(3) @ Rt # Shape: [3, 4]
        
    else:
        raise ValueError("camera_type must be 'affine' or 'projective'")

    return P
