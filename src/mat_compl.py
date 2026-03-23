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
        o       : (F,)      offset per frame
        W_final : (3F, P)   completed matrix (removed rows/cols filled with NaN)
        M_full  : (3F, P)   rank-4 approximation (removed rows/cols filled with NaN)
        mask_out: (3F, P)   final observation mask
    """
    F_orig, P_orig = lam.shape
    device         = lam.device
    dtype          = lam.dtype

    # Active index trackers (boolean over originals)
    active_cols   = torch.ones(P_orig,      dtype=torch.bool, device=device)
    active_frames = torch.ones(F_orig,      dtype=torch.bool, device=device)
    active_rows   = torch.ones(3 * F_orig,  dtype=torch.bool, device=device)  # derived

    # Working views — will be re-sliced in place
    lam_w    = lam.clone()
    tracks_w = tracks.clone()
    mask_w   = mask.clone()

    d = torch.ones(F_orig,  device=device, dtype=dtype)
    o = torch.zeros(F_orig, device=device, dtype=dtype)

    offset_history = []
    eye_r          = ridge * torch.eye(rank, device=device, dtype=dtype)

    # --- SVD initialisation ---
    lam3_w   = lam_w.repeat_interleave(3, dim=0)
    W_init   = lam3_w * tracks_w
    col_mean = torch.nanmean(W_init, dim=0)
    W_filled = torch.where(mask_w, W_init,
                           col_mean.unsqueeze(0).expand_as(W_init))

    Ui, Si, Vhi = torch.linalg.svd(W_filled, full_matrices=False)
    U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()   # (3F, rank)
    V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()   # (P,  rank)
    M = U @ V.T

    best = (float('inf'), d.clone(), o.clone(), M.clone(),
            active_cols.clone(), active_frames.clone())

    for it in range(iters):
        F_w = lam_w.shape[0]
        P_w = lam_w.shape[1]

        lam3_w   = lam_w.repeat_interleave(3, dim=0)
        d3       = d.repeat_interleave(3)
        o3       = o.repeat_interleave(3)
        W_scaled = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
        W_filled = torch.where(mask_w, W_scaled, torch.zeros_like(W_scaled))

        # ---- Outlier removal after warm-up ----
        if it > 4:
            cell_res    = (W_filled - M) ** 2
            cell_res_fp = cell_res.reshape(F_w, 3, P_w).max(dim=1).values  # (F_w, P_w)

            threshold   = torch.quantile(cell_res_fp, 0.9)
            #threshold = 10000
            bad_fp      = cell_res_fp > threshold                           # (F_w, P_w)

            # --- Columns: bad in too many frames ---
            bad_col_count = bad_fp.sum(dim=0)                               # (P_w,)
            remove_cols   = bad_col_count > (F_w - (rank + 4))
            keep_cols     = ~remove_cols

            # --- Rows: bad in too many points ---
            bad_row_count = bad_fp.sum(dim=1)                               # (F_w,)
            remove_frames = bad_row_count > (P_w - (rank + 4))
            keep_frames   = ~remove_frames

            # Per-entry masking for the rest
            mask_w = mask_w & ~bad_fp.repeat_interleave(3, dim=0)

            # Under-observed points
            obs_per_point = mask_w[0::3].sum(dim=0)
            keep_cols     = keep_cols & (obs_per_point >= rank)

            # Under-observed frames
            obs_per_frame = mask_w[0::3].sum(dim=1)
            keep_frames   = keep_frames & (obs_per_frame >= rank)

            n_rem_cols   = (~keep_cols).sum().item()
            n_rem_frames = (~keep_frames).sum().item()

            if n_rem_cols > 0 or n_rem_frames > 0:
                print(f"iter {it:3d} | removing {n_rem_cols} cols, "
                      f"{n_rem_frames} frames "
                      f"→ ({keep_frames.sum().item()} frames, "
                      f"{keep_cols.sum().item()} points remaining)")

                # Expand frame keep mask to 3-row blocks
                keep_rows = keep_frames.repeat_interleave(3)

                # Slice everything consistently
                lam_w    = lam_w[keep_frames][:, keep_cols]
                tracks_w = tracks_w[keep_rows][:, keep_cols]
                mask_w   = mask_w[keep_rows][:, keep_cols]
                W_filled = W_filled[keep_rows][:, keep_cols]
                U        = U[keep_rows]                      # (3F_new, rank)
                V        = V[keep_cols]                      # (P_new,  rank)
                M        = M[keep_rows][:, keep_cols]

                # Update global trackers
                active_cols[active_cols.clone()]     = keep_cols
                active_frames[active_frames.clone()] = keep_frames
                active_rows = active_frames.repeat_interleave(3)

                # Trim d, o to surviving frames
                d = d[keep_frames]
                o = o[keep_frames]

        mask_f = mask_w.float()
        F_w    = lam_w.shape[0]
        P_w    = lam_w.shape[1]

        # ---- ALS: update U ----
        A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_r
        b_U = (mask_f * W_filled) @ V
        U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

        # ---- ALS: update V ----
        A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r
        b_V = (mask_f * W_filled).T @ U
        V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

        M = U @ V.T

        # ---- Affine calibration ----
        d, o = _update_affine_ortho(
            tracks_w[0::3], tracks_w[1::3], lam_w, M, mask=mask_w[0::3]
        )
        o = o - o[0]
        d = torch.ones_like(d)
        offset_history.append(o.clone())

        # ---- Convergence ----
        lam3_w   = lam_w.repeat_interleave(3, dim=0)
        d3       = d.repeat_interleave(3)
        o3       = o.repeat_interleave(3)
        W_scaled = d3[:, None] * (lam3_w + o3[:, None]) * tracks_w
        rho      = (W_scaled - M)[mask_w].norm().item()

        if rho < best[0] - tol:
            best = (rho, d.clone(), o.clone(), M.clone(),
                    active_cols.clone(), active_frames.clone())

    _, d, o, M_best, best_cols, best_frames = best
    best_rows = best_frames.repeat_interleave(3)

    # ---- Scatter back into full-size tensors ----
    W_final  = torch.full((3 * F_orig, P_orig), float('nan'), device=device, dtype=dtype)
    M_full   = torch.full((3 * F_orig, P_orig), float('nan'), device=device, dtype=dtype)
    mask_out = torch.zeros(3 * F_orig, P_orig,  dtype=torch.bool, device=device)
    o_full   = torch.zeros(F_orig,              device=device, dtype=dtype)

    # Final W for surviving entries
    lam3_w   = lam_w.repeat_interleave(3, dim=0)
    d3       = d.repeat_interleave(3)
    o3       = o.repeat_interleave(3)
    W_obs    = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
    W_comp   = torch.where(mask_w, W_obs, M_best)

    rows = best_rows.nonzero(as_tuple=True)[0]   # integer indices for scatter
    cols = best_cols.nonzero(as_tuple=True)[0]

    W_final[rows[:, None], cols[None, :]]  = W_comp
    M_full[rows[:, None],  cols[None, :]]  = M_best
    mask_out[rows[:, None], cols[None, :]] = mask_w

    o_full[best_frames] = o

    # ---- Diagnostics ----
    import matplotlib.pyplot as plt

    if offset_history:
        history_tensor = torch.stack(offset_history)         # (iters, F_surviving)
        plt.figure(figsize=(10, 4))
        for f in range(history_tensor.shape[1]):
            plt.plot(history_tensor[:, f].cpu().numpy(), alpha=0.6)
        plt.title("Offset evolution per frame")
        plt.xlabel("Iteration")
        plt.ylabel("Offset value")
        plt.grid(True, alpha=0.3)
        plt.show()

    cell_res    = (W_comp - M_best) ** 2
    cell_res_fp = cell_res.reshape(lam_w.shape[0], 3, lam_w.shape[1]).max(dim=1).values
    plt.figure(figsize=(14, 5))
    plt.imshow(cell_res_fp.cpu().numpy(), aspect='auto', cmap='hot_r', interpolation='none')
    plt.colorbar()
    plt.title("Per frame-point residual (surviving entries)")
    plt.xlabel("Point")
    plt.ylabel("Frame")
    plt.show()

    return o_full, W_final, M_full, mask_out.float(), active_frames, active_cols




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
