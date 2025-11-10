import torch
import numpy as np
from tqdm import tqdm

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

    return Motion, Shape, T

def proj_stiefel(Wo):
    U, S, Vh = torch.linalg.svd(Wo, full_matrices=False)
    c = S.mean()
    return c * U @ Vh