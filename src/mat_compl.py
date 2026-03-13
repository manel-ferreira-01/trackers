import torch
from tqdm import tqdm
import numpy as np

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
