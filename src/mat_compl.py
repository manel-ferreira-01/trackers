import torch
from tqdm import tqdm
import numpy as np
from src.ortho_factorization import _update_affine_ortho, _update_affine_ortho_lstsq
from src.projective_factorization import projective_factorization_fast

def alternating_matrix_completion(M_obs, rank=4, max_iters=100, tol=1e-6, verbose=False, device='cpu', mode=0, lam=1e-3):

    if type(M_obs) is np.ndarray:
        M_obs = torch.tensor(M_obs, dtype=torch.float32)

    M_obs = M_obs.to(device)
    m, n = M_obs.shape
    mask = ~torch.isnan(M_obs)
    M_filled = torch.nan_to_num(M_obs, nan=0.0)  # replace NaN with 0 for matmul
    eye_rank = lam * torch.eye(rank, device=device)

    if mode:
        U = torch.randn(m, rank, device=device)
        V = torch.randn(n, rank, device=device)
    else:
        new_obs = M_obs.clone()
        col_means = torch.nanmean(M_obs, dim=0)
        inds = torch.isnan(new_obs)
        new_obs[inds] = col_means.expand(m, n)[inds]
        global_mean = torch.mean(new_obs)
        new_obs_centered = new_obs - global_mean
        U_svd, S_svd, V_svd = torch.linalg.svd(new_obs_centered, full_matrices=False)
        U = U_svd[:, :rank] * S_svd[:rank].sqrt()
        V = V_svd[:rank, :].T * S_svd[:rank].sqrt()

    error_list = []
    pbar = tqdm(range(max_iters))

    for iteration in pbar:
        # --- Solve for U (vectorized over all rows) ---
        # V_sub.T @ V_sub per row: (m, rank, rank)
        # mask: (m, n), V: (n, rank)
        mask_f = mask.float()  # (m, n)

        # A[i] = sum_j mask[i,j] * V[j]^T V[j] + lam*I
        # = V.T @ diag(mask[i]) @ V + lam*I
        A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_rank  # (m, rank, rank)

        # b[i] = sum_j mask[i,j] * M[i,j] * V[j]
        # = (mask * M_filled)[i] @ V
        b_U = (mask_f * M_filled) @ V  # (m, rank)

        U = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)  # (m, rank)

        # --- Solve for V (vectorized over all columns) ---
        # A[j] = sum_i mask[i,j] * U[i]^T U[i] + lam*I
        A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_rank  # (n, rank, rank)

        # b[j] = sum_i mask[i,j] * M[i,j] * U[i]
        b_V = (mask_f * M_filled).T @ U  # (n, rank)

        V = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)  # (n, rank)

        # RMSE on observed entries
        M_hat = U @ V.T
        diff = (M_hat - M_obs)[mask]
        error = torch.norm(diff) / torch.sqrt(mask.sum().float())
        error_list.append(error.item())

        if error < tol:
            break
        if iteration > 0 and abs(error_list[-1] - error_list[-2]) / (error_list[-2] + 1e-10) < tol:
            break

        pbar.set_postfix({'RMSE': f'{error.item():.12f}'})

    return U @ V.T, torch.tensor(error_list)


def incremental_matrix_completion(
    M_obs, rank=4, max_iters=100, tol=1e-6, verbose=False, device='cpu',
    mode=0, num_row_blocks=8, num_col_blocks=8, shuffle_blocks=False
):
    """
    Incremental Alternating Least Squares for low-rank matrix completion using PyTorch.
    """

    if type(M_obs) is np.ndarray:
        M_obs = torch.tensor(M_obs, dtype=torch.float32)

    M_obs = M_obs.to(device)
    m, n = M_obs.shape
    mask = ~torch.isnan(M_obs)

    if mode:
        U = torch.randn(m, rank, device=device)
        V = torch.randn(n, rank, device=device)
    else:
        new_obs = M_obs.clone()
        col_means = torch.nanmean(M_obs, dim=0)
        inds = torch.isnan(new_obs)
        new_obs[inds] = col_means.expand(m, n)[inds]

        global_mean = torch.mean(new_obs)
        new_obs_centered = new_obs - global_mean

        U_svd, S_svd, V_svd = torch.linalg.svd(new_obs_centered, full_matrices=False)
        U = U_svd[:, :rank] * S_svd[:rank].sqrt()
        V = V_svd[:rank, :].T * S_svd[:rank].sqrt()

    row_blocks = torch.chunk(torch.arange(m), num_row_blocks)
    col_blocks = torch.chunk(torch.arange(n), num_col_blocks)

    error_list = []

    pbar = tqdm(range(max_iters))
    for iteration in pbar:
        if shuffle_blocks:
            row_blocks = list(row_blocks)
            col_blocks = list(col_blocks)
            np.random.shuffle(row_blocks)
            np.random.shuffle(col_blocks)

        # Update U block by block
        for block_rows in row_blocks:
            for i in block_rows:
                idx = mask[i]
                if idx.any():
                    V_sub = V[idx]
                    M_sub = M_obs[i, idx]
                    sol = torch.linalg.lstsq(V_sub, M_sub.unsqueeze(1)).solution
                    U[i] = sol[:rank].squeeze()

        # Update V block by block
        for block_cols in col_blocks:
            for j in block_cols:
                idx = mask[:, j]
                if idx.any():
                    U_sub = U[idx]
                    M_sub = M_obs[idx, j]
                    sol = torch.linalg.lstsq(U_sub, M_sub.unsqueeze(1)).solution
                    V[j] = sol[:rank].squeeze()

        M_hat = U @ V.T
        diff = (M_hat - M_obs)[mask]
        error = torch.norm(diff) / torch.sqrt(mask.sum())
        error_list.append(error.item())

        if verbose:
            print(f"Iteration {iteration+1}: RMSE = {error.item():.6f}")
        if error < tol:
            break
        if iteration > 0 and abs(error.item() - error_list[-2]) < tol:
            break

        pbar.set_postfix({'RMSE': f'{error.item():.10f}'})

    return U @ V.T, torch.tensor(error_list)


def calibrate_with_completion(tracks, lam, mask, rank=4, iters=100, tol=1e-6, ridge=1e-3):
    """
    Jointly estimates per-frame affine depth calibration (scale + offset)
    and completes missing entries, such that (d*lam + o) * tracks is rank-4.

    Args:
        tracks : (3F, P) pixel tracks [x,y,1] interleaved, NaN where missing
        lam    : (F, P)  monocular depths
        mask   : (3F, P) bool, True where observed
        rank   : target rank
        iters  : max iterations
        tol    : convergence threshold
        ridge  : ALS regularization

    Returns:
        d, o   : (F,) scale and offset per frame
        W      : (3F, P) completed + calibrated matrix
        M      : (3F, P) rank-4 approximation
    """
    F, P   = lam.shape
    device = lam.device
    dtype  = lam.dtype

    x = tracks[0::3]   # (F, P)
    y = tracks[1::3]   # (F, P)

    # --- Initialize d, o ---
    d = torch.ones(F,  device=device, dtype=dtype)
    o = torch.zeros(F, device=device, dtype=dtype)

    offset_history = []
    scales_history = []

    mask_3F  = mask                                          # (3F, P) bool
    mask_f   = mask_3F.float()                               # (3F, P) float
    eye_r    = ridge * torch.eye(rank, device=device, dtype=dtype)

    # --- Initialize U, V from mean-imputed matrix ---
    lam3     = lam.repeat_interleave(3, dim=0)               # (3F, P)
    W_init   = lam3 * tracks
    col_mean = torch.nanmean(W_init, dim=0)
    #W_filled = torch.where(mask_3F, W_init, col_mean.unsqueeze(0).expand_as(W_init))
    W_filled = torch.where(mask_3F, W_init, torch.zeros_like(W_init)) # start with zeros for missing

    Ui, Si, Vhi = torch.linalg.svd(W_filled, full_matrices=False)
    U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()       # (3F, rank)
    V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()       # (P,  rank)
    M = U @ V.T                                              # (3F, P)

    best     = (float('inf'), d.clone(), o.clone(), M.clone())

    for it in range(iters):

        # ---- 1. Build calibrated W (observed entries only) ----
        d3       = d.repeat_interleave(3)                    # (3F,)
        o3       = o.repeat_interleave(3)                    # (3F,)
        W_scaled = (d3[:, None] * lam3 + o3[:, None]) * tracks   # (3F, P), NaN where missing

        # Fill missing with current low-rank prediction
        W_filled = torch.where(mask_3F, W_scaled, torch.zeros_like(W_scaled))        # (3F, P)

        # ---- 2. ALS completion: one pass ----
        # Update U
        A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_r   # (3F, rank, rank)
        b_U = (mask_f * W_filled) @ V                                 # (3F, rank)
        U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1) # (3F, rank)

        # Update V
        A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r   # (P, rank, rank)
        b_V = (mask_f * W_filled).T @ U                               # (P, rank)
        V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1) # (P, rank)

        M   = U @ V.T                                        # (3F, P)
        
        #W_filled = torch.nan_to_num(W_scaled, nan=30.0)
        #U, S, Vh = torch.linalg.svd(W_filled, full_matrices=False)
        #M = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        
        # ---- 3. Offset + scale solve against M ----
        # TODO:CHECK THIS
        d, o = _update_affine_ortho(x, y, lam, M, mask=mask_3F[0::3])
        #d, o = _update_affine_ortho_lstsq(x,y, lam, M, mask=mask_3F[0::3])
        #print("d",d)
        o = o-o[0]                                      # zero-mean offset
        offset_history.append(o.clone())
        d    = torch.ones_like(d)                            # scale absorbed into shape

        # ---- 4. Convergence ----
        d3       = d.repeat_interleave(3)
        o3       = o.repeat_interleave(3)
        W_scaled = d3[:, None] *  (lam3 + o3[:, None]) * tracks

        if 0:
            W_filled = torch.where(mask_3F, W_scaled, torch.zeros_like(W_scaled))        # (3F, P)
            motion, shape, tvec, scales = projective_factorization_fast(W_filled)
            d = scales
            d3 = d.repeat_interleave(3)
            W_scaled = (d3[:, None] * lam3 + o3[:, None])
            print("scales",scales)
        
        # Only measure residual on observed entries
        diff = (W_scaled - M)
        rho = diff[mask_3F].norm().item()


        best = (rho, d.clone(), o.clone(), M.clone())
        if rho < best[0] - tol:
            pass

    _, d, o, M = best

    # Final completed matrix
    d3       = d.repeat_interleave(3)
    o3       = o.repeat_interleave(3)
    W_final  = (d3[:, None] * lam3 + o3[:, None]) * tracks
    W_final  = torch.where(mask_3F, W_final, M)             # fill missing with rank-4

    if 1:
        motion, shape, tvec, scales = projective_factorization_fast(W_final)
        scales = scales / scales.max()
        d = scales
        d3 = d.repeat_interleave(3)
        W_final = (d3[:, None] * lam3 + o3[:, None]) * tracks
        W_final  = torch.where(mask_3F, W_final, M)             # fill missing with rank-4



    # Plot offset history at the end
    import matplotlib.pyplot as plt
    history_tensor = torch.stack(offset_history).cpu().numpy()  # (iters, F)
    plt.figure(figsize=(10, 4))
    for f in range(F):
        plt.plot(history_tensor[:, f], label=f"Frame {f}")
    plt.title("Offset evolution per frame")
    plt.xlabel("Iteration")
    plt.ylabel("Offset value")
    #plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot scales history at the end
    #scales_tensor = torch.stack(scales_history).cpu().numpy()  # (iters, F)
    #plt.figure(figsize=(10, 4))
    #for f in range(F):
    #    plt.plot(scales_tensor[:, f], label=f"Frame {f}")
    #plt.title("Scale evolution per frame")
    #plt.xlabel("Iteration")
    #plt.ylabel("Scale value")
    #plt.legend()
    #plt.grid(True, alpha=0.3)
    #plt.show()

    return o, W_final, M


def calibrate_with_completion_claude(tracks, lam, mask, rank=4, iters=100, tol=1e-6, ridge=1e-3):
    F, P   = lam.shape
    device = lam.device
    dtype  = lam.dtype

    x = tracks[0::3]
    y = tracks[1::3]

    mask_F  = mask[0::3]          # (F, P) — per point mask
    mask_f  = mask.float()        # (3F, P)

    # impute lam nans with per-frame mean
    lam_filled = lam.clone()
    lam_filled[torch.isnan(lam)] = torch.nanmean(lam)

    o  = torch.zeros(F, device=device, dtype=dtype)
    
    # Initialize tracks_filled — fill missing with column mean
    tracks_filled = tracks.clone()
    col_means = torch.nanmean(tracks, dim=0)
    tracks_filled[torch.isnan(tracks)] = col_means.expand_as(tracks)[torch.isnan(tracks)]

    # Initialize UV from first scaled matrix
    Z = (lam_filled + o[:, None]).repeat_interleave(3, dim=0) * tracks_filled
    print(torch.isnan(Z).any().sum())
    Ui, Si, Vhi = torch.linalg.svd(Z, full_matrices=False)
    U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()
    V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()
    M = U @ V.T

    eye_r  = ridge * torch.eye(rank, device=device, dtype=dtype)
    best   = (float('inf'), o.clone(), M.clone())
    offset_history = []

    for it in range(iters):

        # ---- 1. Update tracks_filled: fill missing W entries from M and current o ----
        # M_fp = (lam_fp + o_f) * W_fp  =>  W_fp = M_fp / (lam_fp + o_f)
        denom        = (lam_filled + o[:, None]).repeat_interleave(3, dim=0)  # (3F, P)
        W_from_M     = M / denom.clamp(min=1e-8)                              # (3F, P)
        tracks_filled = torch.where(mask, tracks, W_from_M)                   # fill missing

        # ---- 2. Build scaled matrix ----
        Z        = (lam_filled + o[:, None]).repeat_interleave(3, dim=0) * tracks_filled
        Z_obs    = torch.where(mask, Z, M)                                    # observed=data, missing=M

        # ---- 3. ALS completion on Z (one pass) ----
        A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_r
        b_U = (mask_f * Z_obs) @ V
        U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

        A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r
        b_V = (mask_f * Z_obs).T @ U
        V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

        M   = U @ V.T                                                         # (3F, P)

        # Use corrected M for offset solve
        _, o = _update_affine_ortho(x, y, lam_filled, M, mask=mask_F)
        #o = o - o[0]
        offset_history.append(o.clone())

        # ---- 4.5 Update missing lam entries from M ----
        lam_from_M   = M[2::3]                                    # (F, P) = lam + o
        lam_updated  = torch.where(mask_F, lam_filled, lam_from_M - o[:, None])

        # Then use lam_updated instead of lam_filled for next iteration
        #lam_filled = lam_updated

        # ---- 5. Convergence on observed entries ----
        Z_scaled = (lam_filled + o[:, None]).repeat_interleave(3, dim=0) * tracks_filled
        rho      = (Z_scaled - M)[mask].norm().item()

        #if rho < best[0]:
        best = (rho, o.clone(), M.clone())

    _, o, M = best

    # Final: fill missing tracks from M
    denom         = (lam_filled + o[:, None]).repeat_interleave(3, dim=0)
    W_from_M      = M / denom.clamp(min=1e-8)
    tracks_final  = torch.where(mask, tracks, W_from_M)
    tracks_final[torch.isnan(tracks_final)] = 0.0

    # Plot
    import matplotlib.pyplot as plt
    history = torch.stack(offset_history).cpu().numpy()
    plt.figure(figsize=(10, 4))
    for f in range(F):
        plt.plot(history[:, f], label=f"Frame {f}")
    plt.title("Offset evolution")
    plt.xlabel("Iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return o, tracks_final, M


def check_visibility(mask_F, rank=4):
    """
    mask_F: (F, P) bool — True where observed
    """
    obs_per_frame = mask_F.sum(dim=1)   # (F,) — points per frame
    obs_per_point = mask_F.sum(dim=0)   # (P,) — frames per point

    bad_frames = (obs_per_frame < rank).nonzero().squeeze()
    bad_points = (obs_per_point < rank).nonzero().squeeze()

    print(f"Obs per frame: min={obs_per_frame.min().item()} max={obs_per_frame.max().item()}")
    print(f"Obs per point: min={obs_per_point.min().item()} max={obs_per_point.max().item()}")
    
    if len(bad_frames.shape) > 0 and bad_frames.numel() > 0:
        print(f"Frames with < {rank} observations: {bad_frames.tolist()}")
    if len(bad_points.shape) > 0 and bad_points.numel() > 0:
        print(f"Points visible in < {rank} frames: {bad_points.numel()}")

    return obs_per_frame, obs_per_point

# And filter to only keep valid points/frames
def filter_visibility(tracks, lam, mask, rank=4):
    mask_F       = mask[0::3]                            # (F, P)
    valid_frames = torch.ones(mask_F.shape[0], dtype=torch.bool)
    valid_points = torch.ones(mask_F.shape[1], dtype=torch.bool)

    for _ in range(100):
        obs_per_frame = mask_F.sum(dim=1)
        obs_per_point = mask_F.sum(dim=0)

        new_vf = obs_per_frame >= rank
        new_vp = obs_per_point >= rank

        if new_vf.all() and new_vp.all():
            break

        mask_F = mask_F[new_vf][:, new_vp]
        tracks = tracks[new_vf.repeat_interleave(3)][:, new_vp]
        lam    = lam[new_vf][:, new_vp]

        # accumulate valid indices
        valid_frames[valid_frames.clone()] = new_vf
        valid_points[valid_points.clone()] = new_vp

    print(f"After filtering: F={mask_F.shape[0]}, P={mask_F.shape[1]}")
    return tracks, lam, mask_F.repeat_interleave(3, dim=0), valid_frames, valid_points
