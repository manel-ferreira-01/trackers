import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.ortho_factorization import calibrate_orthographic
from src.projective_factorization import projective_factorization_fast
from src.mat_compl import alternating_matrix_completion

def run_reconstruction(
    W_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    num_outer_iterations: int = 300,
    num_inner_iterations: int = 1,
    num_calib_steps: int = 1,
    outer_tol: float = 1e-3,
    inner_tol: float = 1e-4,
    calib_rank: int = 4,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the projective factorization + orthographic calibration reconstruction pipeline.

    Args:
        W_mat:                  Visibility/weight matrix, shape (3F, N).
        lambda_mat:             Per-frame depth scale matrix, shape (F, N).
        num_outer_iterations:   Max outer loop iterations (offset calibration).
        num_inner_iterations:   Max inner loop iterations (scale update via projective factorization).
        num_calib_steps:        Number of calibration steps per outer iteration.
        outer_tol:              Convergence threshold for global offset stability.
        inner_tol:              Convergence threshold for scale stability.
        calib_rank:             Rank passed to calibrate_orthographic.
        plot:                   Whether to show the offset history plot.
        verbose:                Whether to show the tqdm progress bar.

    Returns:
        A dict with keys:
            "motion"         - Motion matrix from final factorization, shape (3F, R).
            "shape"          - Shape matrix, shape (R, N).
            "tvec"           - Translation vectors, shape (3F,).
            "scales"         - Per-frame scale estimates, shape (F,).
            "cam_lists"      - List of F camera matrices P = [R|t], each (3, 4).
            "total_offsets"  - Final accumulated depth offsets, shape (F,).
            "current_scales" - Final accumulated scale factors, shape (F,).
            "offset_history" - Tensor of offset snapshots, shape (total_steps, F).
    """
    device = lambda_mat.device
    F = lambda_mat.shape[0]
    N = W_mat.shape[1]

    # --- 1. Initialization ---
    Lambda = lambda_mat.clone().repeat_interleave(3, 0)

    total_offsets = torch.zeros(F, device=device)
    current_scales = torch.ones(F, device=device)
    offset_history = []

    # --- 2. Optimization Loops ---
    pbar = tqdm(range(num_outer_iterations), disable=not verbose)
    scale_change = torch.tensor(0.0, device=device)

    for outer_i in pbar:
        prev_offsets = total_offsets.clone()

        off_map = total_offsets.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
        scl_map = current_scales.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
        current_Lambda = (Lambda + off_map) / scl_map

        # A. Orthographic Calibration
        _, delta_snapshots, _, _ = calibrate_orthographic(
            W_mat,
            current_Lambda[0::3],
            K=torch.eye(3, device=device),
            iters=num_calib_steps,
            rank=calib_rank,
            tol=1e-6,
        )

        # B. Record snapshots
        for snap in delta_snapshots:
            offset_history.append(total_offsets + snap)

        # C. Update global offset accumulator
        total_offsets = total_offsets + delta_snapshots[-1]

        # D. Projective Factorization Inner Loop
        for inner_i in range(num_inner_iterations):
            prev_inner_scales = current_scales.clone()

            off_map = total_offsets.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
            scl_map = current_scales.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
            iter_Lambda = (Lambda + off_map) / scl_map

            motion, shape, tvec, scales = projective_factorization_fast(iter_Lambda * W_mat)

            scales = scales / scales.max()
            current_scales = current_scales * scales

            scale_change = torch.norm(current_scales - prev_inner_scales) / (
                torch.norm(prev_inner_scales) + 1e-9
            )

            if scale_change < inner_tol:
                break

        offset_change = torch.norm(total_offsets - prev_offsets) / (
            torch.norm(prev_offsets) + 1e-9
        )

        if verbose:
            pbar.set_postfix({
                "off_Δ": f"{offset_change.item():.2e}",
                "scl_Δ": f"{scale_change.item():.2e}",
                #"inner_iters": f"{inner_i + 1}",
            })

        if offset_change < outer_tol:
            if verbose:
                print(f"\n>>> Converged at outer iter {outer_i} (off_Δ={offset_change:.2e})")
            break

    # --- 3. Final Reconstruction ---
    final_off_map = total_offsets.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
    final_scl_map = current_scales.unsqueeze(1).repeat_interleave(3, 0).expand(-1, N)
    final_Lambda = (Lambda + final_off_map) / final_scl_map

    motion, shape, tvec, scales = projective_factorization_fast(final_Lambda * W_mat)

    if verbose:
        print(f"Final scales mean: {scales.mean().item():.4f}")

    # --- 4. Camera Matrices ---
    R1_inv = motion[:3, :3].t()
    t1_est = tvec[:3]

    cam_lists = []
    for frame in range(F):
        Mi = motion[frame * 3: (frame + 1) * 3, :]
        ti = tvec[frame * 3: (frame + 1) * 3]
        Mi_new = Mi @ R1_inv
        ti_new = ti - (Mi_new @ t1_est)
        cam_lists.append(torch.cat((Mi_new, ti_new), dim=1))

    aligned_shape = motion[:3, :3] @ shape + t1_est


    # --- 5. Plot ---
    if plot and offset_history:
        history_tensor = torch.stack(offset_history).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(history_tensor, "-x", label=[f"Frame {i}" for i in range(F)])
        plt.title("Accumulated Offsets (Snapshot-based progression)")
        plt.xlabel("Total Calibration Steps")
        plt.ylabel("Total Offset Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "motion": motion,
        "shape": aligned_shape,
        "tvec": tvec,
        "scales": scales,
        "cam_lists": cam_lists,
        "total_offsets": total_offsets,
        "current_scales": current_scales,
        "offset_history": torch.stack(offset_history) if offset_history else torch.empty(0),
    }
