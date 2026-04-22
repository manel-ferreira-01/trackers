import torch


def column_growing_init(
    W_obs: torch.Tensor,
    mask_w: torch.Tensor,
    rank: int = 4,
    anchor_iters: int = 30,
    ridge: float = 1e-3,
    min_frame_obs: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Warm-start (U, V) for ALS matrix completion.

    The standard mean-fill + global SVD initializer is poor when entries are
    sparse (~50%) with a temporal-band pattern: mean-fill from distant frames
    corrupts the imputed values, and the resulting SVD mixes unrelated subspaces.

    Fix
    ---
    1.  Sort columns by observation count (desc).  The top-K columns are the
        most universally observed — they anchor the row subspace U because they
        provide real (not imputed) constraints for every frame.
    2.  Find the smallest K such that every frame has >= min_frame_obs real
        observations in those K columns.  This sub-problem is dense by
        construction; imputation here is trustworthy.
    3.  Run ALS on this (F, K) sub-problem to get a clean U.  No gauge
        fragmentation: one problem, one ALS, one subspace.
    4.  With U fixed, solve V for ALL P columns in one ALS pass.  V rows for
        sparse columns are determined by the same U, so the full factorization
        is gauge-consistent.

    The (U, V) pair is a drop-in replacement for the SVD init in
    `calibrate_with_completion` / `alternating_matrix_completion`.

    Args:
        W_obs          : (3F, P)  observed entries, zeros where missing.
        mask_w         : (3F, P)  bool, True where observed.
        rank           : target rank.
        anchor_iters   : ALS iterations on the dense anchor sub-problem.
        ridge          : Tikhonov regularisation for normal equations.
        min_frame_obs  : minimum observations per frame in the anchor set.
                         Defaults to 2*rank.

    Returns:
        U : (3F, rank)
        V : (P,  rank)
    """
    device = W_obs.device
    dtype  = W_obs.dtype
    n_rows, P = W_obs.shape

    if min_frame_obs is None:
        min_frame_obs = 2 * rank

    eye_r = ridge * torch.eye(rank, device=device, dtype=dtype)

    # --- 1. Sort columns by observation count ---
    col_obs   = mask_w[0::3].sum(dim=0).float()         # (P,)
    col_order = torch.argsort(col_obs, descending=True)  # (P,)

    # --- 2. Binary search for smallest anchor set covering all frames ---
    lo, hi = rank + 1, P
    while lo < hi:
        mid = (lo + hi) // 2
        frame_obs = mask_w[0::3][:, col_order[:mid]].sum(dim=1)
        if frame_obs.min() >= min_frame_obs:
            hi = mid
        else:
            lo = mid + 1
    K = lo
    anchor_cols = col_order[:K]

    frame_obs_anchor = mask_w[0::3][:, anchor_cols].sum(dim=1)
    #print(f"[column_growing_init] anchor: K={K}/{P} cols "
    #      f"| frame obs {frame_obs_anchor.min().item()}..{frame_obs_anchor.max().item()} "
    #      f"| density={mask_w[:, anchor_cols].float().mean():.3f}")

    # --- 3. ALS on the dense anchor sub-problem ---
    W_anc    = W_obs[:, anchor_cols]
    mask_anc = mask_w[:, anchor_cols].float()

    col_mean = (W_anc * mask_anc).sum(0) / mask_anc.sum(0).clamp(min=1)
    W_filled = torch.where(mask_anc.bool(), W_anc,
                           col_mean.unsqueeze(0).expand_as(W_anc))
    Ui, Si, Vhi = torch.linalg.svd(W_filled, full_matrices=False)
    U     = (Ui[:, :rank] * Si[:rank].sqrt()).contiguous()
    V_anc = (Vhi[:rank].T * Si[:rank].sqrt()).contiguous()

    for _ in range(anchor_iters):
        A_U = torch.einsum('ij,jk,jl->ikl', mask_anc, V_anc, V_anc) + eye_r
        b_U = (mask_anc * W_anc) @ V_anc
        U   = torch.linalg.solve(A_U, b_U.unsqueeze(-1)).squeeze(-1)

        A_V = torch.einsum('ij,ik,il->jkl', mask_anc, U, U) + eye_r
        b_V = (mask_anc * W_anc).T @ U
        V_anc = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        diff = (U @ V_anc.T - W_anc)[mask_anc.bool()]
        #print(f"[column_growing_init] anchor RMSE={((diff**2).mean().sqrt().item()):.5f}")

    # --- 4. Single global V-solve from the anchor U (gauge-consistent) ---
    mask_f = mask_w.float()
    A_V = torch.einsum('ij,ik,il->jkl', mask_f, U, U) + eye_r
    b_V = (mask_f * W_obs).T @ U
    V   = torch.linalg.solve(A_V, b_V.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        diff_full = (U @ V.T - W_obs)[mask_w]
        #print(f"[column_growing_init] full RMSE after V-solve={((diff_full**2).mean().sqrt().item()):.5f}")

    return U, V