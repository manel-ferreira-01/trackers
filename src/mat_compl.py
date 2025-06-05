import numpy as np
from tqdm import tqdm

def alternating_matrix_completion(M_obs, rank=5, max_iters=100, tol=1e-4, verbose=False):
    """
    Alternating Least Squares for low-rank matrix completion.
    
    Parameters:
        M_obs: np.ndarray
            Partially observed matrix with np.nan for missing values.
        rank: int
            Desired rank of the completed matrix.
        max_iters: int
            Maximum number of iterations.
        tol: float
            Convergence tolerance based on Frobenius norm.
        verbose: bool
            Whether to print progress.
    
    Returns:
        M_completed: np.ndarray
            The completed matrix.
    """
    m, n = M_obs.shape
    mask = ~np.isnan(M_obs)

    # Random initialization
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)

    #save error over iterations
    error_list = []

    for iteration in tqdm(range(max_iters)):
        # Fix V, solve for U
        for i in range(m):
            
            idx = mask[i, :]
            if np.any(idx): # if there is at least one observed entry
                V_sub = V[idx, :]
                M_sub = M_obs[i, idx]
                U[i, :] = np.linalg.lstsq(V_sub, M_sub, rcond=None)[0]

        # Fix U, solve for V
        for j in range(n):
            idx = mask[:, j]
            if np.any(idx):
                U_sub = U[idx, :]
                M_sub = M_obs[idx, j]
                V[j, :] = np.linalg.lstsq(U_sub, M_sub, rcond=None)[0]

        # Compute reconstruction error on observed entries
        M_hat = U @ V.T
        error = np.linalg.norm((M_hat - M_obs)[mask]) / np.sqrt(np.sum(mask))
        error_list.append(error)

        if verbose:
            pass
            #print(f"Iteration {iteration+1}: RMSE = {error:.6f}")
        if error < tol:
            break

        # if error remains constant, break
        if iteration > 0 and abs(error - error_list[-2]) < tol:
            break

    return U @ V.T, np.array(error_list)
