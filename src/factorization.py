import torch
import numpy as np

def marques_factorization(obs_mat: torch.Tensor):
    """
    obs_mat: torch.Tensor of shape (2*num_frame, num_features)
    Returns:
        M: motion matrix (2*num_frames x 3)
        S: shape matrix (3 x num_features)
        tvecs: translation vectors per frame (num_frames x 2)
        alphas: scaling parameters per frame (num_frames,)
    """

    if type(obs_mat) is np.ndarray:
        obs_mat = torch.tensor(obs_mat, dtype=torch.float32)
    device = obs_mat.device
    dtype = obs_mat.dtype
    Xs = obs_mat[0::2, :]
    Ys = obs_mat[1::2, :]

    # Center observations
    tvecs = torch.stack([Xs.mean(dim=1), Ys.mean(dim=1)], dim=1)  # (num_frames, 2)
    Xs = Xs - Xs.mean(dim=1, keepdim=True)
    Ys = Ys - Ys.mean(dim=1, keepdim=True)
    obs_mat_centered = torch.cat([Xs, Ys], dim=0)

    # SVD decomposition
    U, S, Vh = torch.linalg.svd(obs_mat_centered, full_matrices=True)
    V = Vh.T

    S_root = torch.sqrt(S[0:3])
    M_hat = U[:, 0:3] * S_root
    S_hat = S_root[:, None] * V[:, 0:3].T

    num_frames = obs_mat.shape[0] // 2
    A_rows = []
    b = []

    for f in range(num_frames):
        A_rows.append(constraint_torch(M_hat[f], M_hat[f]))
        b.append(1.0)
        A_rows.append(constraint_torch(M_hat[num_frames + f], M_hat[num_frames + f]))
        b.append(1.0)
        A_rows.append(constraint_torch(M_hat[f], M_hat[num_frames + f]))
        b.append(0.0)

    A = torch.stack(A_rows, dim=0)  # (3*num_frames, 6)
    b = torch.tensor(b, dtype=dtype, device=device).unsqueeze(1)  # (3*num_frames, 1)

    # Build the alpha constraints
    mat = torch.kron(torch.eye(num_frames, dtype=dtype, device=device), torch.tensor([[-1.0, -1.0, 0.0]], dtype=dtype, device=device).reshape(3,1))
    A = torch.cat([A, mat], dim=1)

    # Solve homogeneous system A x = 0 via SVD
    _, _, Vh_full = torch.linalg.svd(A)
    l = Vh_full[-1]
    l = l / l[6]  # Normalize by l[6]
    alphas = l[6:]

    # Build Q matrix and compute its Cholesky factor L
    Q = torch.tensor([
        [l[0], l[1], l[2]],
        [l[1], l[3], l[4]],
        [l[2], l[4], l[5]]
    ], dtype=dtype, device=device)

    L = torch.linalg.cholesky(Q)

    M = M_hat @ L
    S = torch.linalg.inv(L) @ S_hat

    return M, S, tvecs, alphas


def constraint_torch(m1, m2):
    return torch.tensor([
        m1[0] * m2[0],
        m1[0] * m2[1] + m1[1] * m2[0],
        m1[0] * m2[2] + m1[2] * m2[0],
        m1[1] * m2[1],
        m1[1] * m2[2] + m1[2] * m2[1],
        m1[2] * m2[2]
    ], dtype=m1.dtype, device=m1.device)