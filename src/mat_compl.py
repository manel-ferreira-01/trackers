import torch
from tqdm import tqdm
import numpy as np
from src.ortho_factorization import _update_affine_ortho, _update_affine_ortho_lstsq
from src.projective_factorization import projective_factorization_fast
from src.column_growing_init import column_growing_init

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

def ransac_subspace(W_filled, mask_w, rank=4, n_iters=100, 
                    threshold=None, min_sample=20):
    """
    W_filled : (3F, P) — completed/observed matrix, zeros where missing
    mask_w   : (3F, P) bool
    Returns  : inlier_cols (P,) bool mask
    """
    F3, P = W_filled.shape
    device = W_filled.device

    if threshold is None:
        # set threshold as median absolute column norm * factor
        col_norms = W_filled.norm(dim=0)  # (P,)
        threshold = col_norms.median() * 0.005

    best_inliers = torch.zeros(P, dtype=torch.bool, device=device)
    best_count = 0

    for _ in range(n_iters):
        # 1. Sample minimal set of columns
        sample_idx = torch.randperm(P, device=device)[:min_sample]
        W_sample = W_filled[:, sample_idx]  # (3F, min_sample)

        # 2. Fit rank-4 subspace from sample — use LEFT singular vectors
        U_s, _, _ = torch.linalg.svd(W_sample, full_matrices=False)
        V_basis = U_s[:, :rank]  # (3F, rank) — column space basis

        # 3. Project all columns onto subspace and measure residual
        W_proj = V_basis @ (V_basis.T @ W_filled)  # (3F, P)
        residuals = (W_filled - W_proj).norm(dim=0)  # (P,)

        # 4. Count inliers
        inliers = residuals < threshold
        count = inliers.sum().item()

        if count > best_count:
            best_count = count
            best_inliers = inliers

    return best_inliers, torch.abs(W_filled - W_proj)

def calibrate_with_completion(tracks, lam, mask, rank=4, iters=100, tol=1e-4, ridge=1e-10,
                              offset_mode="normalize", removal_iters=(10, 20, 30, 40),
                              min_obs=2, completion="als", svt_tau=500, svt_step=1.0):
    """
    Jointly estimates per-frame affine depth calibration (scale + offset)
    and completes missing entries, such that (d*lam + o) * tracks is rank-4.

    Args:
        tracks      : (3F, P) pixel tracks [x,y,1] interleaved, NaN where missing
        lam         : (F, P)  monocular depths
        mask        : (3F, P) bool, True where observed
        rank        : target rank
        iters       : max iterations
        tol         : convergence threshold
        ridge       : ALS regularization
        offset_mode : how to handle the per-frame offset after estimation:
                        "estimate"   – use raw estimated o
                        "normalize"  – estimate o, then subtract o[0] (default)
                        "zero"       – always keep o = 0 (no offset)

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
    frame_scales = torch.ones(F_orig, device=device, dtype=dtype)  # accumulated per-frame scales

    offset_history = []
    eye_r          = ridge * torch.eye(rank, device=device, dtype=dtype)
    cell_res_fp    = None

    # --- SVD initialisation ---
    lam3_w   = lam_w.repeat_interleave(3, dim=0)
    W_init   = lam3_w * tracks_w
    W_init_z = torch.where(mask_w, W_init, torch.zeros_like(W_init))  # zeros for missing


    if 1:
        col_mean = torch.nanmean(W_init, dim=0)
        W_filled = torch.where(mask_w, W_init, col_mean.unsqueeze(0).expand_as(W_init))
        Ui, Si, Vhi = torch.linalg.svd(W_filled, full_matrices=False)
        U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()
        V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()
        M = U @ V.T

    else:
        U, V = column_growing_init(W_init_z, mask_w, rank=rank, ridge=ridge)
        M = U @ V.T

    prev_rho = float('inf')
    Y_svt = torch.zeros_like(M)   # SVT dual variable (persistent across iters)

    for it in range(iters):
        F_w = lam_w.shape[0]
        P_w = lam_w.shape[1]

        lam3_w   = lam_w.repeat_interleave(3, dim=0)
        d3       = d.repeat_interleave(3)
        o3       = o.repeat_interleave(3)
        W_scaled = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
        W_filled = torch.where(mask_w, W_scaled, torch.zeros_like(W_scaled))
        W_filled = torch.where(mask_w, W_scaled, M)
        #W_filled = torch.where(mask_w, W_scaled, torch.zeros_like(W_scaled))

        # ---- Outlier removal at fixed iterations ----
        if removal_iters and it in removal_iters:

            if 0:
                res_A = (W_filled - M) ** 2
                # --- Option B: project W_filled onto V subspace ---
                Qv    = torch.linalg.svd(V, full_matrices=False)[0]   # (P_w, rank)
                W_proj_V = (W_filled @ Qv) @ Qv.T
                res_B = (W_filled - W_proj_V) ** 2

                # --- Option C: project W_filled onto U subspace ---
                Qu    = torch.linalg.svd(U, full_matrices=False)[0]   # (3F_w, rank)
                W_proj_U = Qu @ (Qu.T @ W_filled)
                res_C = (W_filled - W_proj_U) ** 2

                # --- Switch here ---
                cell_res    = res_A   # <-- swap to res_B or res_C to test
                cell_res_fp = cell_res.reshape(F_w, 3, P_w).max(dim=1).values

                #threshold   = torch.quantile(cell_res_fp, 0.9)
                #bad_fp      = cell_res_fp > threshold
                threshold = torch.quantile(cell_res_fp, 0.90)
                abs_floor = cell_res_fp.median() * 100.0   # only remove if residual is 5x median
                threshold = torch.maximum(threshold, abs_floor) 
                bad_fp = cell_res_fp > abs_floor


                # --- Columns: bad in too many frames ---
                bad_col_count = bad_fp.sum(dim=0)
                col_thresh    = max(F_w - 2, 1)          # avoid 0 threshold with few frames
                remove_cols   = bad_col_count >= col_thresh
                keep_cols     = ~remove_cols

                # --- Rows: bad in too many points ---
                bad_row_count = bad_fp.sum(dim=1)
                row_thresh    = max(P_w - 2, 1)          # avoid 0 threshold with few points
                remove_frames = bad_row_count >= row_thresh
                keep_frames   = ~remove_frames

                # Per-entry masking only for entries NOT in removed rows/cols
                # so that mask_w stays consistent with V/U sizes until slicing
                bad_fp_entry = bad_fp.clone()
                bad_fp_entry[~keep_frames] = False   # these rows will be removed entirely
                bad_fp_entry[:, ~keep_cols] = False  # these cols will be removed entirely
                mask_w = mask_w & ~bad_fp_entry.repeat_interleave(3, dim=0)
            elif 0:
                # per-cell residual magnitude
                res = (W_filled - M).abs()
                cell_res_fp = res.reshape(F_w, 3, P_w).max(dim=1).values  # (F_w, P_w)

                # --- MAD threshold per column (robust to outlier inflation) ---
                # For each point (column), compute median and MAD across frames
                col_median = cell_res_fp.median(dim=0).values          # (P_w,)
                col_mad    = (cell_res_fp - col_median).abs().median(dim=0).values  # (P_w,)
                col_mad    = col_mad.clamp(min=1e-6)
                # robust Z-score per entry
                robust_z   = (cell_res_fp - col_median) / (1.4826 * col_mad)  # (F_w, P_w)
                bad_fp     = robust_z > 3.0   # flag entries > 3 sigma

                # --- Column removal: bad in more than X% of frames ---
                bad_col_frac = bad_fp.float().mean(dim=0)   # (P_w,)
                remove_cols  = bad_col_frac > 0.8           # remove if bad in >50% of frames
                keep_cols    = ~remove_cols

                # --- Frame removal: bad in more than X% of points ---
                bad_row_frac = bad_fp.float().mean(dim=1)   # (F_w,)
                remove_frames = bad_row_frac > 0.8
                keep_frames   = ~remove_frames

                # per-entry masking for entries not in removed rows/cols
                bad_fp_entry = bad_fp.clone()
                bad_fp_entry[~keep_frames] = False
                bad_fp_entry[:, ~keep_cols] = False
                mask_w = mask_w & ~bad_fp_entry.repeat_interleave(3, dim=0)

            else:

                inlier_cols, cell_res_fp = ransac_subspace(
                    W_filled, mask_w, rank=rank, 
                    n_iters=100, min_sample=10,
                )
                keep_cols   = inlier_cols
                keep_frames = torch.ones(F_w, dtype=torch.bool, device=device)  # RANSAC on cols only

                bad_fp_entry = torch.zeros_like(mask_w[0::3], dtype=torch.bool)  # (F_w, P_w)
                bad_fp_entry[~keep_frames] = False   # these rows will be removed entirely
                bad_fp_entry[:, ~keep_cols] = False  # these cols will be removed entirely
                mask_w = mask_w & ~bad_fp_entry.repeat_interleave(3, dim=0)
                
                cell_res_fp = cell_res_fp * mask_w  # only consider observed entries for diagnostics


            # Under-observed after per-entry masking
            # clamp to actual frame/point count so 2-frame windows aren't emptied
            obs_per_point = mask_w[0::3].sum(dim=0)
            keep_cols     = keep_cols & (obs_per_point >= min(min_obs, F_w))

            obs_per_frame = mask_w[0::3].sum(dim=1)
            keep_frames   = keep_frames & (obs_per_frame >= min(min_obs, P_w))

            n_rem_cols   = (~keep_cols).sum().item()
            n_rem_frames = (~keep_frames).sum().item()

            if n_rem_cols > 0 or n_rem_frames > 0:
                print(f"iter {it:3d} | removing {n_rem_cols} cols, "
                      f"{n_rem_frames} frames "
                      f"→ ({keep_frames.sum().item()} frames, "
                      f"{keep_cols.sum().item()} points remaining)")

                keep_rows = keep_frames.repeat_interleave(3)

                lam_w    = lam_w[keep_frames][:, keep_cols]
                tracks_w = tracks_w[keep_rows][:, keep_cols]
                mask_w   = mask_w[keep_rows][:, keep_cols]
                U        = U[keep_rows]
                V        = V[keep_cols]

                active_cols[active_cols.clone()]     = keep_cols
                active_frames[active_frames.clone()] = keep_frames
                active_rows = active_frames.repeat_interleave(3)

                d = d[keep_frames]
                o = o[keep_frames]

                lam3_w   = lam_w.repeat_interleave(3, dim=0)
                d3       = d.repeat_interleave(3)
                o3       = o.repeat_interleave(3)
                W_scaled = (d3[:, None] * lam3_w + o3[:, None]) * tracks_w
                W_filled = torch.where(mask_w, W_scaled, torch.zeros_like(W_scaled))
                M        = U @ V.T
                Y_svt    = Y_svt[keep_rows][:, keep_cols]

        if it % 10 == 0:
           #print("iter",it, mask_w.float().mean())
           pass

        mask_f = mask_w.float()
        F_w    = lam_w.shape[0]
        P_w    = lam_w.shape[1]

        if completion == "als":
            # ---- ALS: update U ----
            A_U = torch.einsum('ij,jk,jl->ikl', mask_f, V, V) + eye_r
            b_U = (mask_f * W_filled) @ V
            U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

            # ---- ALS: update V ----
            A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r
            b_V = (mask_f * W_filled).T @ U
            V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

        else:  # completion == "svt"
            # ---- SVT: one step of nuclear norm minimization ----
            # dual ascent on observed entries
            for i in range(10):
                Y_svt = Y_svt + svt_step * torch.where(mask_w, W_filled - M, torch.zeros_like(M))

                # soft-threshold singular values
                U_s, s_s, Vh_s = torch.linalg.svd(Y_svt, full_matrices=False)
                s_thresh = torch.clamp(s_s - svt_tau, min=0.0)
                M = (U_s * s_thresh) @ Vh_s

                # re-extract rank-r factors for downstream code (affine calibration, outlier removal)
                U = (U_s[:, :rank] * s_thresh[:rank].sqrt()).contiguous()
                V = (Vh_s[:rank].T  * s_thresh[:rank].sqrt()).contiguous()

        if completion == "als":
            M = U @ V.T

        # ---- Affine calibration ----
        if it > 40:
            d, o = _update_affine_ortho(
                tracks_w[0::3], tracks_w[1::3], lam_w, M, mask=mask_w[0::3]
            )
            if offset_mode == "normalize":
                o = o - o[0]
            elif offset_mode == "zero":
                o = torch.zeros_like(o)
        else:
            o = torch.zeros_like(o)

        #d = d / d.mean()
        d = torch.ones_like(d)

        # ---- Per-frame scale + focal correction from singular values ----
        offset_history.append(o.clone())

        # ---- Convergence ----
        lam3_w   = lam_w.repeat_interleave(3, dim=0)
        d3       = d.repeat_interleave(3)
        o3       = o.repeat_interleave(3)
        W_scaled = d3[:, None] * (lam3_w + o3[:, None]) * tracks_w
        
        rho      = (W_scaled - M)[mask_w].norm().item()

        if it > 40 and abs(prev_rho - rho) < tol and (len(offset_history) < 2 or torch.allclose(offset_history[-1], offset_history[-2], atol=tol)):
            #print(f"Converged at iteration {it} with step {abs(prev_rho - rho):.6f} and stable offset")
            break
        prev_rho = rho


    M_best = M
    best_cols, best_frames = active_cols, active_frames
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
    if 0:
        import matplotlib.pyplot as plt

        if offset_history:
            # Pad each snapshot to F_orig length with NaN before stacking
            history_padded = torch.full((len(offset_history), F_orig), float('nan'),
                                        device=device, dtype=dtype)
            af = active_frames.clone()  # final surviving frames mask
            # Rebuild frame mask at each iteration is expensive — instead just
            # place each snapshot into the surviving slots cumulatively.
            # We know final active_frames; earlier snapshots are subsets of it.
            # Simplest: just right-align by length and warn if ambiguous.
            for i, snap in enumerate(offset_history):
                history_padded[i, :snap.shape[0]] = snap   # left-align by current frame order

            history_tensor = history_padded.cpu().numpy()
            plt.figure(figsize=(10, 4))
            for f in range(F_orig):
                vals = history_tensor[:, f]
                if not np.all(np.isnan(vals)):
                    plt.plot(vals, alpha=0.6, label=f"Frame {f}")
            plt.title("Offset evolution per frame")
            plt.xlabel("Iteration")
            plt.ylabel("Offset value")
            #plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()


        #cell_res    = (W_comp - M_best) ** 2
        #cell_res_fp = cell_res.reshape(lam_w.shape[0], 3, lam_w.shape[1]).max(dim=1).values
        if cell_res_fp is not None:
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
    mask_F       = mask[0::3]
    valid_frames = torch.ones(mask_F.shape[0], dtype=torch.bool)
    valid_points = torch.ones(mask_F.shape[1], dtype=torch.bool)

    for _ in range(100):
        F, P = mask_F.shape
        frame_thresh = min(rank, P)
        point_thresh = min(rank, F)

        obs_per_frame = mask_F.sum(dim=1)
        obs_per_point = mask_F.sum(dim=0)

        new_vf = obs_per_frame >= frame_thresh
        new_vp = obs_per_point >= point_thresh

        if new_vf.all() and new_vp.all():
            break

        mask_F = mask_F[new_vf][:, new_vp]
        tracks = tracks[new_vf.repeat_interleave(3)][:, new_vp]
        lam    = lam[new_vf][:, new_vp]

        valid_frames[valid_frames.clone()] = new_vf
        valid_points[valid_points.clone()] = new_vp

    #print(f"After filtering: F={mask_F.shape[0]}, P={mask_F.shape[1]}")
    return tracks, lam, mask_F.repeat_interleave(3, dim=0), valid_frames, valid_points



def windowed_init(W_obs, mask_w, rank=4, window_size=25, overlap=10, ridge=1e-3, iters=30):
    """
    Warm-start initialization for ALS by factorizing overlapping frame windows.

    W_obs  : (3F, P) — observed measurement matrix (zeros where missing, not NaN)
    mask_w : (3F, P) — bool, True where observed
    Returns: W_filled (3F, P) — dense matrix suitable for SVD initialization
    """
    device = W_obs.device
    dtype  = W_obs.dtype
    n_rows, P = W_obs.shape
    F = n_rows // 3

    # Accumulate weighted completions
    W_acc    = torch.zeros_like(W_obs)
    W_weight = torch.zeros(n_rows, P, device=device, dtype=dtype)

    step = max(window_size - overlap, 1)
    starts = list(range(0, F, step))
    # Ensure last window reaches the end
    if starts[-1] + window_size < F:
        starts.append(F - window_size)

    eye_r = ridge * torch.eye(rank, device=device, dtype=dtype)

    for start in starts:
        end = min(start + window_size, F)
        row_slice = slice(start * 3, end * 3)

        W_sub   = W_obs[row_slice]     # (3*win, P)
        mask_sub = mask_w[row_slice]   # (3*win, P)

        # Keep only columns with ≥2 observations in this window
        col_obs  = mask_sub[0::3].sum(0)   # (P,) — use x-rows as proxy
        good_col = col_obs >= 2
        if good_col.sum() < rank:
            continue

        Ws = W_sub[:, good_col]
        ms = mask_sub[:, good_col].float()

        # Quick SVD init for this window
        col_mean = (Ws * ms).sum(0) / ms.sum(0).clamp(min=1)
        Wf = torch.where(ms.bool(), Ws, col_mean.unsqueeze(0).expand_as(Ws))
        Ui, Si, Vhi = torch.linalg.svd(Wf, full_matrices=False)
        U = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()
        V = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()

        # Few ALS iterations on this window only
        for _ in range(iters):
            A_U = torch.einsum('ij,jk,jl->ikl', ms, V, V) + eye_r
            b_U = (ms * Ws) @ V
            U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

            A_V = torch.einsum('ij,ik,il->jkl', ms, U, U) + eye_r
            b_V = (ms * Ws).T @ U
            V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

        W_comp = U @ V.T  # (3*win, |good_col|)

        # Weight by number of observations per column in this window
        w = col_obs[good_col].float()  # (|good_col|,)
        W_acc[row_slice][:, good_col]    += W_comp * w.unsqueeze(0)
        W_weight[row_slice][:, good_col] += w.unsqueeze(0)

    # Normalize overlap regions; fall back to 0 where no window covered
    W_filled = W_acc / W_weight.clamp(min=1)

    # Always trust real observations over window completions
    W_filled = torch.where(mask_w, W_obs, W_filled)

    return W_filled

