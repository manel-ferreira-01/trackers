import torch
from src.mat_compl import calibrate_with_completion, filter_visibility, check_visibility
from src.projective_factorization import projective_factorization_fast
from auxiliar.depth_tensor_viz import k3d_3d_plot


def run_projective_reconstruction(
    W_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    iters: int = 100,
    num_scale_iters: int = 4,
    rank: int = 4,
    seed: int = 42,
    offset_mode: str = "normalize",
    removal_iters: tuple = (10, 20, 30, 40),
    plot: bool = True,
) -> dict:
    """
    Full projective reconstruction pipeline: filter visibility, complete the
    measurement matrix, run projective factorization with alpha scaling, and
    return cameras + shape.

    Args:
        W_mat:            Observation matrix (3F, P), NaNs for missing entries.
        lambda_mat:       Depth scale matrix (F, P), NaNs for missing entries.
        iters:            Iterations for calibrate_with_completion.
        num_scale_iters:  Alpha-scaling refinement iterations.
        rank:             Rank used in filter_visibility.
        seed:             Torch random seed for reproducibility.
        offset_mode:      "estimate", "normalize" (default), or "zero" — passed to calibrate_with_completion.
        plot:             Whether to call k3d_3d_plot at the end.

    Returns dict with keys:
        cam_lists         list of (3, 4) camera matrices
        aligned_shape     (3, P) shape aligned to first camera
        final_W           (3F, P) normalized observation matrix
        final_lam         (F, P) final depth scales
        compl_W_lam       (3F, P) completed W*lambda matrix (after alpha scaling)
        current_scales    (F,) per-frame scale factors
        offsets           (F,) per-frame offsets from completion
        vf                (F_orig,) bool mask of surviving frames in original indexing
        vp                (P_orig,) bool mask of surviving points in original indexing
        mask_f            (3F_surv, P_surv) final observation mask
        surviving_frames  (F_filtered,) bool mask
        surviving_cols    (P_filtered,) bool mask
    """
    torch.manual_seed(seed)

    # --- Build NaN matrices from mask ---
    mask = ~torch.isfinite(lambda_mat)
    W_mat_nan = W_mat.clone()
    W_mat_nan[mask.repeat_interleave(3, dim=0)] = float('nan')
    lambda_mat_nan = lambda_mat.clone()
    lambda_mat_nan[mask] = float('nan')

    # --- Filter visibility ---
    tracks_f, lam_f, mask_f, vf, vp = filter_visibility(
        W_mat_nan, lambda_mat_nan,
        (~mask).repeat_interleave(3, dim=0), rank=rank,
    )
    check_visibility(mask_f[0::3])

    nan_pct = (1 - mask_f.float().mean()) * 100
    print(f"NaNs after filter_visibility: {nan_pct:.2f}%")

    # --- Matrix completion ---
    o, compl_W_lam, M, mask_f, surviving_frames, surviving_cols = calibrate_with_completion(
        tracks_f, lam_f, mask_f, iters=iters, offset_mode=offset_mode, removal_iters=removal_iters,
    )

    # --- Align vf/vp to surviving frames/cols ---
    vp_indices = vp.nonzero(as_tuple=True)[0]
    vp[vp_indices[~surviving_cols]] = False

    vf_indices = vf.nonzero(as_tuple=True)[0]
    vf[vf_indices[~surviving_frames]] = False

    # --- Drop removed rows/cols from working matrices ---
    sel_rows = surviving_frames.repeat_interleave(3)
    compl_W_lam = compl_W_lam[sel_rows][:, surviving_cols]
    M           = M[sel_rows][:, surviving_cols]
    mask_f      = mask_f[sel_rows][:, surviving_cols]

    # --- Projective factorization with alpha scaling ---
    motion, shape, tvec, alphas = projective_factorization_fast(compl_W_lam)
    print("Alphas before scaling:", alphas)

    compl_W_lam_scaled = compl_W_lam.clone()
    current_scales = torch.ones_like(alphas)

    for i in range(num_scale_iters):
        motion, shape, tvec, alphas = projective_factorization_fast(
            compl_W_lam_scaled / current_scales.repeat_interleave(3, 0)[:, None]
        )
        current_scales *= alphas / alphas.max()
        print(f"  iter {i + 1} alphas: {alphas}")

    # --- Final matrices ---
    final_W_lam = compl_W_lam_scaled / current_scales.repeat_interleave(3, 0)[:, None]
    final_lam = final_W_lam[2::3]
    final_W   = final_W_lam / final_lam.repeat_interleave(3, dim=0)

    print("current_scales:", current_scales)
    print("Final alphas:  ", alphas)

    # --- Build camera list aligned to first camera ---
    R1_inv = motion[:3, :3].t()
    t1_est = tvec[:3]
    cam_lists = []
    for f in range(motion.shape[0] // 3):
        Mi = motion[f * 3: (f + 1) * 3, :]
        ti = tvec[f * 3: (f + 1) * 3]
        Mi_new = Mi @ R1_inv
        cam_lists.append(torch.cat((Mi_new, ti - Mi_new @ t1_est), dim=1))

    aligned_shape = motion[:3, :3] @ shape + t1_est

    if plot:
        k3d_3d_plot(aligned_shape, camera_input=cam_lists)

    return dict(
        cam_lists=cam_lists,
        aligned_shape=aligned_shape,
        final_W=final_W,
        final_lam=final_lam,
        compl_W_lam=final_W_lam,
        current_scales=current_scales,
        offsets=o,
        vf=vf,
        vp=vp,
        mask_f=mask_f,
        surviving_frames=surviving_frames,
        surviving_cols=surviving_cols,
    )
