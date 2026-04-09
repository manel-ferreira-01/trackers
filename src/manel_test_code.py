import torch
import matplotlib.pyplot as plt


def projective_joint_imputation(
    tracks_f: torch.Tensor,
    lambda_raw: torch.Tensor,
    mask_f: torch.Tensor,
    iter_outer: int = 50,
    iter_inner: int = 10,
    stop_outer: float = 1e-5,
    stop_inner: float = 1e-3,
    verbose: bool = False,
    plot_evolution: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint alternating factorization + depth offset correction.

    Pipeline per inner iteration:
        1. Update Shape   (lstsq given Motion, on depth-normalised W)
        2. Update Motion  (lstsq given Shape)
        3. Project each frame's 3x3 rotation block onto scaled-rotation manifold
           via SVD; use mean singular value s_f as per-frame scale.
        4. Update offset_f — closed form minimising x/y reprojection error only:
             s = Σ_p m_p [(x_p·Mx_p + y_p·My_p) - λ_p·(x_p²+y_p²)]
                 / Σ_p m_p (x_p²+y_p²)
           alpha is a gauge DOF (scaling depth rows is absorbed by Motion)
           so we fix alpha=1 and only fit the additive offset.
        5. Re-build W_lam: x/y from tracks, depth = (lambda_raw + offset) / depth_scale
        6. Impute missing entries with Motion @ Shape + T

    Depth normalisation: each frame's depth row is divided by its per-frame
    median observed depth before factorization, so x/y and depth rows have
    comparable magnitude in the lstsq. The scale is stored and undone on output.

    Args:
        tracks_f:   (3F, P) normalized homogeneous observations, NaN where missing.
                    Rows 3f, 3f+1 are x/y bearing vectors; row 3f+2 is homogeneous 1.
        lambda_raw: (F, P) raw MDE depth scales, NaN where missing.
        mask_f:     (3F, P) bool, True where observed.
        iter_outer: Max outer iterations.
        iter_inner: Max inner (Motion/Shape/offset) iterations.
        stop_outer: Convergence threshold on imputed-entry change.
        stop_inner: Convergence threshold on Motion update norm.
        verbose:    Print per-iteration diagnostics.
        plot_evolution: Show offset and convergence plot after completion.

    Returns:
        W_lam_filled:  (3F, P) completed W*lambda (in original depth units).
        lambda_cal:    (F, P)  calibrated depth = lambda_raw + offset_f.
        offsets:       (F,)    per-frame depth offsets.
        Motion:        (3F, 4) final motion matrix (in normalised depth space).
        Shape:         (4, P)  final shape matrix  (in normalised depth space).
    """
    device = tracks_f.device
    dtype  = tracks_f.dtype
    ThreeF, P = tracks_f.shape
    F = ThreeF // 3

    # depth row indices: 2, 5, 8, ...
    depth_rows = torch.arange(2, ThreeF, 3, device=device)

    # ------------------------------------------------------------------ #
    # Per-frame depth normalisation                                        #
    # Divide each depth row by its median observed depth so that depth     #
    # rows have scale ~1, matching x/y bearing rows.                       #
    # Store depth_scale (F,) to undo on output.                            #
    # ------------------------------------------------------------------ #
    m_d = mask_f[depth_rows].float()                              # (F, P)
    lam_vals = lambda_raw.nan_to_num(0.0)
    # median over observed points per frame; fall back to 1 if no obs
    depth_scale = torch.zeros(F, device=device, dtype=dtype)
    for f in range(F):
        obs = m_d[f].bool()
        if obs.sum() > 0:
            depth_scale[f] = lam_vals[f, obs].median()
    depth_scale = depth_scale.clamp(min=1e-6)                    # (F,)

    lambda_norm = lambda_raw / depth_scale[:, None]              # (F, P) normalised

    # ------------------------------------------------------------------ #
    # Build initial W_lam (normalised): x/y from tracks, depth normalised #
    # ------------------------------------------------------------------ #
    W_lam = tracks_f.clone()
    W_lam[depth_rows] = lambda_norm
    W_lam = torch.where(mask_f, W_lam,
                        torch.tensor(float('nan'), device=device, dtype=dtype))

    # ------------------------------------------------------------------ #
    # Initialise offset (in normalised depth space)                       #
    # ------------------------------------------------------------------ #
    offsets_norm = torch.zeros(F, device=device, dtype=dtype)

    # history
    offset_history = []
    error_history  = []

    # ------------------------------------------------------------------ #
    # Mean-impute for initial working copy                                 #
    # ------------------------------------------------------------------ #
    row_sum   = W_lam.nan_to_num(0.0).sum(dim=1)
    row_count = mask_f.float().sum(dim=1).clamp(min=1)
    row_mean  = (row_sum / row_count).unsqueeze(1)
    W = torch.where(mask_f, W_lam, row_mean.expand_as(W_lam))

    # Translation proxy: prefer a fully-observed column
    obs_counts = mask_f.sum(dim=0)
    full_cols  = (obs_counts == ThreeF).nonzero(as_tuple=True)[0]
    T_vec = (W[:, full_cols[0]].unsqueeze(1).clone() if len(full_cols) > 0
             else W.mean(dim=1, keepdim=True))

    # ------------------------------------------------------------------ #
    # Bootstrap Motion/Shape from SVD                                      #
    # ------------------------------------------------------------------ #
    W_c = W - T_vec
    U, S_vals, Vh = torch.linalg.svd(W_c, full_matrices=False)
    S_root = torch.sqrt(S_vals[:4])
    Motion = (U[:, :4] * S_root).contiguous()         # (3F, 4)
    Shape  = (S_root[:, None] * Vh[:4]).contiguous()  # (4, P)

    # ------------------------------------------------------------------ #
    # Outer loop                                                           #
    # ------------------------------------------------------------------ #
    error_outer = float('inf')
    outer_iter  = 0
    W_hat_full  = Motion @ Shape + T_vec

    while error_outer > stop_outer and outer_iter < iter_outer:

        W_c = W - T_vec

        # ---- Inner loop -------------------------------------------- #
        error_inner = float('inf')
        inner_iter  = 0

        while error_inner > stop_inner and inner_iter < iter_inner:

            # 1. Update Shape given Motion
            Shape = torch.linalg.lstsq(Motion, W_c).solution          # (4, P)

            # 2. Update Motion given Shape
            Motion_prev = Motion.clone()
            Motion = torch.linalg.lstsq(Shape.T, W_c.T).solution.T   # (3F, 4)

            # 3. Project rotation block onto scaled-rotation manifold.
            #    M_f[:, :3] = s_f * R_f  (SVD nearest scaled rotation).
            #    Projective column [:, 3] is left untouched.
            Mf       = Motion.reshape(F, 3, 4)
            Mf_R     = Mf[:, :, :3]                              # (F, 3, 3)
            Mf_p     = Mf[:, :, 3:4]                             # (F, 3, 1)
            U_m, S_m, Vh_m = torch.linalg.svd(Mf_R)
            # fix reflections
            dets     = torch.linalg.det(U_m @ Vh_m)             # (F,)
            flip     = torch.ones(F, 1, device=device, dtype=dtype)
            flip[dets < 0] = -1.0
            Vh_m     = Vh_m * flip.unsqueeze(2)
            s_f      = S_m.mean(dim=1)                           # (F,) per-frame scale
            Mf_R_proj = (U_m * s_f[:, None, None]) @ Vh_m       # (F, 3, 3) = s_f * R_f
            Motion   = torch.cat([Mf_R_proj, Mf_p], dim=2).reshape(ThreeF, 4)

            # 4. Update offset from x/y reprojection (closed form).
            #    Minimise: Σ_p m_p [(x_p(λ_p+s) - Mx_p)² + (y_p(λ_p+s) - My_p)²]
            #    → s = Σ_p m_p [(x_p·Mx_p + y_p·My_p) - λ_p·(x_p²+y_p²)]
            #          / Σ_p m_p (x_p²+y_p²)
            #    All quantities in normalised depth space.
            pred  = Motion @ Shape + T_vec                        # (3F, P)
            Mx_S  = pred[0::3]                                    # (F, P)
            My_S  = pred[1::3]

            x_  = torch.nan_to_num(tracks_f[0::3],  nan=0.0)    # (F, P)
            y_  = torch.nan_to_num(tracks_f[1::3],  nan=0.0)
            L_  = torch.nan_to_num(lambda_norm,      nan=0.0)    # normalised
            Mx_ = torch.nan_to_num(Mx_S,             nan=0.0)
            My_ = torch.nan_to_num(My_S,             nan=0.0)
            m_xy = mask_f[0::3].float()                          # (F, P)

            X2Y2 = x_**2 + y_**2
            numer       = (m_xy * (x_ * Mx_ + y_ * My_ - L_ * X2Y2)).sum(1)
            denom       = (m_xy * X2Y2).sum(1).clamp(min=1e-6)
            offsets_norm = numer / denom                          # (F,) normalised

            # 5. Re-build W (normalised): depth = lambda_norm + offset_norm
            lambda_cal_norm = (lambda_norm + offsets_norm[:, None]).clamp(min=1e-3)
            W_lam_new = tracks_f.clone()
            W_lam_new[depth_rows] = lambda_cal_norm
            W_hat_full = Motion @ Shape + T_vec
            W = torch.where(mask_f, W_lam_new, W_hat_full)

            # Re-centre
            T_vec = (W[:, full_cols[0]].unsqueeze(1) if len(full_cols) > 0
                     else W.mean(dim=1, keepdim=True))
            W_c = W - T_vec

            error_inner = (torch.norm(Motion - Motion_prev) /
                           torch.norm(Motion_prev).clamp(min=1e-9)).item()
            if verbose:
                sv_spread = (S_m.max(dim=1).values /
                             S_m.min(dim=1).values.clamp(min=1e-9)).mean().item()
                print(f"  inner {inner_iter:3d}  motion_err={error_inner:.2e}  "
                      f"sv_spread={sv_spread:.3f}  "
                      f"offset_norm=[{offsets_norm.min():.3f}, {offsets_norm.max():.3f}]")
            inner_iter += 1

        # ---- Outer convergence on missing entries -------------------- #
        W_hat_full  = Motion @ Shape + T_vec
        missing     = ~mask_f
        n_missing   = missing.float().sum().clamp(min=1)
        error_outer = (torch.norm(W[missing] - W_hat_full[missing]) /
                       n_missing.sqrt()).item()

        if verbose:
            print(f"outer {outer_iter:3d}  impute_err={error_outer:.2e}")

        # Convert offsets back to original depth units for history
        offsets_orig = offsets_norm * depth_scale
        offset_history.append(offsets_orig.detach().cpu().clone())
        error_history.append(error_outer)

        W = torch.where(mask_f, W, W_hat_full)
        T_vec = (W[:, full_cols[0]].unsqueeze(1) if len(full_cols) > 0
                 else W.mean(dim=1, keepdim=True))
        outer_iter += 1

    # ------------------------------------------------------------------ #
    # Final outputs — convert back to original depth units                #
    # ------------------------------------------------------------------ #
    offsets      = offsets_norm * depth_scale                     # (F,) original units
    lambda_cal   = (lambda_raw + offsets[:, None]).clamp(min=1e-3)

    # Observed entries: x/y from tracks, depth = lambda_cal (original units)
    W_obs = tracks_f.clone()
    W_obs[depth_rows] = lambda_cal
    # Missing entries: model prediction converted back to original depth units
    W_lam_model = W_hat_full.clone()
    W_lam_model[depth_rows] = W_hat_full[depth_rows] * depth_scale[:, None]
    W_lam_filled = torch.where(mask_f, W_obs, W_lam_model)

    # ------------------------------------------------------------------ #
    # Optional evolution plot                                             #
    # ------------------------------------------------------------------ #
    if plot_evolution and len(offset_history) > 0:
        offset_hist = torch.stack(offset_history, dim=0).numpy()  # (iters, F)
        iters_range = range(len(error_history))

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax = axes[0]
        for f in range(F):
            ax.plot(iters_range, offset_hist[:, f], label=f"f{f}")
        ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel("offset (original depth units)")
        ax.set_title("Per-frame depth offset — evolution over outer iterations")
        if F <= 12:
            ax.legend(fontsize=7, ncol=min(F, 6), loc='upper right')

        ax = axes[1]
        ax.semilogy(iters_range, error_history, color='steelblue',
                    marker='o', markersize=3)
        ax.set_ylabel("impute err (log)")
        ax.set_xlabel("outer iteration")

        plt.tight_layout()
        plt.show()

    return W_lam_filled, lambda_cal, offsets, Motion, Shape