import torch
from tqdm import tqdm
import numpy as np
def alternating_matrix_completion(M_obs, rank=5, max_iters=100, tol=1e-4, verbose=False, device='cpu'):
    """
    Alternating Least Squares for low-rank matrix completion using PyTorch.

    Parameters:
        M_obs: torch.Tensor
            Partially observed matrix with NaNs for missing values.
        rank: int
            Desired rank of the completed matrix.
        max_iters: int
            Maximum number of iterations.
        tol: float
            Convergence tolerance based on Frobenius norm.
        verbose: bool
            Whether to print progress.
        device: str
            Device to use ('cpu' or 'cuda').

    Returns:
        M_completed: torch.Tensor
            The completed matrix.
        error_list: torch.Tensor
            List of RMSE errors per iteration.
    """

    if type(M_obs) is np.ndarray:
        M_obs = torch.tensor(M_obs, dtype=torch.float32)
        
    M_obs = M_obs.to(device)
    m, n = M_obs.shape
    mask = ~torch.isnan(M_obs)

    # Random initialization
    U = torch.randn(m, rank, device=device)
    V = torch.randn(n, rank, device=device)

    error_list = []

    for iteration in tqdm(range(max_iters)):
        # Fix V, solve for U
        for i in range(m):
            idx = mask[i]
            if idx.any():
                V_sub = V[idx]  # shape: (num_observed, rank)
                M_sub = M_obs[i, idx]  # shape: (num_observed,)
                sol = torch.linalg.lstsq(V_sub, M_sub.unsqueeze(1)).solution
                U[i] = sol[:rank].squeeze()

        # Fix U, solve for V
        for j in range(n):
            idx = mask[:, j]
            if idx.any():
                U_sub = U[idx]
                M_sub = M_obs[idx, j]
                sol = torch.linalg.lstsq(U_sub, M_sub.unsqueeze(1)).solution
                V[j] = sol[:rank].squeeze()

        # Compute RMSE on observed entries
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

    return U @ V.T, torch.tensor(error_list)
