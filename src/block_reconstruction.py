import torch
from dataclasses import dataclass
from typing import Optional
import importlib, src.projective_reconstruction
importlib.reload(src.projective_reconstruction)
from src.projective_reconstruction import run_projective_reconstruction

import matplotlib.pyplot as plt

@dataclass
class BlockResult:
    frame_indices: list          # global frame indices, e.g. [4, 5, 6, 7]
    point_indices: torch.Tensor  # global point indices (P_block,) — after internal filtering
    shape: torch.Tensor          # (3, P_block) in local projective frame
    motion: torch.Tensor         # (3*F_block, 4) local camera matrices
    tvec: torch.Tensor           # (3*F_block, 1) translation vector
    mask_f: torch.Tensor         # (3*F_block, P_block) observation mask
    offsets: torch.Tensor        # (F_block,) per-frame offsets from ALS
    scales: torch.Tensor         # (F_block,) per-frame scales from ALS


# ---------------------------------------------------------------------------
# Stage 1: Partition
# ---------------------------------------------------------------------------

def make_blocks(F: int, block_size: int = 4, overlap: int = 2) -> list[list[int]]:
    """
    Partition F frames into overlapping blocks of size block_size.
    The last block is extended or shifted to always cover frame F-1.

    Returns list of frame index lists, e.g. [[0,1,2,3], [2,3,4,5], ...]
    """
    assert overlap < block_size, "overlap must be less than block_size"
    step = block_size - overlap
    blocks = []
    start = 0
    while start < F:
        end = min(start + block_size, F)
        blocks.append(list(range(start, end)))
        if end == F:
            break
        start += step
    # If last block is smaller than block_size, absorb into previous
    if len(blocks) > 1 and len(blocks[-1]) < max(2, overlap + 1):
        blocks[-2] = sorted(set(blocks[-2]) | set(blocks[-1]))
        blocks.pop()
    return blocks


# ---------------------------------------------------------------------------
# Stage 2: Local reconstruction per block
# ---------------------------------------------------------------------------

def reconstruct_block(
    W_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    frame_ids: list[int],
    global_point_mask: torch.Tensor,
    run_kwargs: dict,
) -> Optional[BlockResult]:
    """
    Extract sub-matrices for frame_ids, run run_projective_reconstruction,
    and return a BlockResult with global point/frame indices restored.
    """
    device = W_mat.device

    row_ids = []
    for f in frame_ids:
        row_ids += [3 * f, 3 * f + 1, 3 * f + 2]
    row_ids = torch.tensor(row_ids, device=device)

    point_ids_global = global_point_mask.nonzero(as_tuple=True)[0]   # (P_sub,)
    sub_W   = W_mat[row_ids][:, point_ids_global]                     # (3*F_b, P_sub)
    sub_lam = lambda_mat[torch.tensor(frame_ids, device=device)][:, point_ids_global]  # (F_b, P_sub)
    
    #plt.imshow(sub_W * sub_lam.repeat_interleave(3, dim=0), aspect='auto', cmap='Blues', interpolation='none')

    obs_per_frame = torch.isfinite(sub_lam).sum(dim=1)
    if obs_per_frame.min() < 4:
        return None

    try:
        result = run_projective_reconstruction(sub_W, sub_lam, plot=False, **run_kwargs)
    except Exception as e:
        print(f"  Block {frame_ids} failed: {e}")
        return None

    # Map surviving points back to global indices
    surviving_local      = result["vp"]                            # (P_sub,) bool
    global_ids_surviving = point_ids_global[surviving_local]       # (P_surv,)

    # Map surviving frames back to global indices
    surviving_frame_mask = result["vf"]                            # (F_b,) bool
    surviving_frame_ids  = [frame_ids[i] for i, ok in enumerate(surviving_frame_mask.tolist()) if ok]

    F_surv = len(surviving_frame_ids)
    if F_surv == 0:
        return None

    cam_list = result["cam_lists"]   # list of F_surv (3,4) tensors
    motion   = torch.stack([c[:, :3] for c in cam_list], dim=0).reshape(3 * F_surv, 3)
    tvec     = torch.stack([c[:, 3]  for c in cam_list], dim=0).reshape(3 * F_surv, 1)
    motion4  = torch.cat([motion, tvec], dim=1)   # (3F_surv, 4)

    return BlockResult(
        frame_indices=surviving_frame_ids,
        point_indices=global_ids_surviving,
        shape=result["aligned_shape"],    # (3, P_surv)
        motion=motion4,                   # (3F_surv, 4)
        tvec=tvec,                        # (3F_surv, 1)
        mask_f=result["mask_f"],          # (3F_surv, P_surv)
        offsets=result["offsets"],        # (F_surv,) ALS offsets
        scales=result["current_scales"],  # (F_surv,) ALS scales
    )


# ---------------------------------------------------------------------------
# Stage 3: Stitching
# ---------------------------------------------------------------------------

def find_common_points(block_a: BlockResult, block_b: BlockResult):
    """
    Returns index tensors (idx_a, idx_b) into each block's point_indices
    for the points that appear in both blocks.
    """
    match = (block_a.point_indices.unsqueeze(1) == block_b.point_indices.unsqueeze(0))
    idx_a, idx_b = match.nonzero(as_tuple=True)
    return idx_a, idx_b


def estimate_H(block_a: BlockResult, block_b: BlockResult, min_points: int = 6) -> Optional[torch.Tensor]:
    """
    Estimate 4x4 projective homography H such that:
        shape_a[:, common] ≈ H @ shape_b_hom[:, common]

    Returns H (4, 4) or None if too few common points.
    """
    idx_a, idx_b = find_common_points(block_a, block_b)
    if idx_a.numel() < min_points:
        print(f"  Too few common points for stitching: {idx_a.numel()} < {min_points}")
        return None

    device = block_a.shape.device
    dtype  = block_a.shape.dtype

    def to_hom(S):   # (3, P) → (4, P)
        return torch.cat([S, torch.ones(1, S.shape[1], device=device, dtype=dtype)], dim=0)

    S_a = to_hom(block_a.shape)[:, idx_a]   # (4, K)
    S_b = to_hom(block_b.shape)[:, idx_b]   # (4, K)

    # Solve S_a = H @ S_b  →  H.T = lstsq(S_b.T, S_a.T)
    H_T, _, _, _ = torch.linalg.lstsq(S_b.T, S_a.T)
    H = H_T.T
    H = H / H[3, 3]

    if torch.linalg.det(H) < 0:
        H = -H

    return H


def consistency_error(block_a: BlockResult, block_b: BlockResult, H_ab: torch.Tensor) -> float:
    """
    For overlap frames, compare block_b cameras (after H alignment) with block_a cameras.
    Returns mean Frobenius distance — a diagnostic of stitching quality.
    """
    overlap_frames = set(block_a.frame_indices) & set(block_b.frame_indices)
    if not overlap_frames:
        return float('nan')

    H_inv = torch.linalg.inv(H_ab)
    errors = []
    for f in overlap_frames:
        ia = block_a.frame_indices.index(f)
        ib = block_b.frame_indices.index(f)
        cam_a         = block_a.motion[ia * 3: ia * 3 + 3]   # (3, 4)
        cam_b         = block_b.motion[ib * 3: ib * 3 + 3]   # (3, 4)
        cam_b_aligned = cam_b @ H_inv                          # (3, 4) in frame A
        errors.append((cam_a - cam_b_aligned).norm().item())

    return float(torch.tensor(errors).mean())


# ---------------------------------------------------------------------------
# Stage 4: Global assembly
# ---------------------------------------------------------------------------

def apply_H_to_block(block: BlockResult, H: torch.Tensor) -> BlockResult:
    """
    Transform a block's shape and motion into a new projective frame via H.
        shape_new  = H @ shape_hom          (dehomogenised)
        motion_new = motion @ H^{-1}        cameras transform contravariantly
    """
    device = block.shape.device
    dtype  = block.shape.dtype
    H_inv  = torch.linalg.inv(H)

    ones      = torch.ones(1, block.shape.shape[1], device=device, dtype=dtype)
    shape_hom = torch.cat([block.shape, ones], dim=0)        # (4, P)
    shape_new_hom = H @ shape_hom                             # (4, P)
    shape_new = shape_new_hom[:3] / shape_new_hom[3].unsqueeze(0)

    F_b           = block.motion.shape[0] // 3
    motion_new    = block.motion.reshape(F_b, 3, 4) @ H_inv.unsqueeze(0)
    motion_new    = motion_new.reshape(F_b * 3, 4)

    return BlockResult(
        frame_indices=block.frame_indices,
        point_indices=block.point_indices,
        shape=shape_new,
        motion=motion_new,
        tvec=block.tvec,
        mask_f=block.mask_f,
        offsets=block.offsets,
        scales=block.scales,
    )


# ---------------------------------------------------------------------------
# Per-frame affine calibration in global frame
# ---------------------------------------------------------------------------

def fit_affine(x: torch.Tensor, y: torch.Tensor):
    """
    Fit y = s * x + o via least squares.
    x, y: (K,) tensors of matched depth values.
    Returns (s, o) scalars.
    """
    A = torch.stack([x, torch.ones_like(x)], dim=1)   # (K, 2)
    sol, _, _, _ = torch.linalg.lstsq(A, y.unsqueeze(1))
    return sol[0, 0], sol[1, 0]


def compute_global_depth_calibration(
    assembled_blocks: list[BlockResult],
    lambda_mat: torch.Tensor,
    cam_lists: list,
    frame_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each surviving global frame, fit a per-frame affine calibration
    (scale, offset) that maps raw MDE depths to the global reconstruction frame.

    Strategy: for each frame f, find all points observed in f that also have
    a 3D position in the global shape. Compute their projective depth under
    the global camera, then fit:
        lambda_global = s * lambda_MDE_raw + o

    Args:
        assembled_blocks : list of BlockResult after stitching
        lambda_mat       : (F_orig, P_orig) raw MDE depth matrix (NaN where missing)
        cam_lists        : list of (3,4) global camera matrices (from assemble_blocks)
        frame_indices    : list of global frame indices corresponding to cam_lists

    Returns:
        global_scales  : (F_surv,) per-frame scale  s[f]
        global_offsets : (F_surv,) per-frame offset o[f]
        — indexed by frame_indices order
    """
    device = lambda_mat.device
    dtype  = lambda_mat.dtype

    # Build global shape lookup: global_point_idx → (3,) tensor
    point_to_xyz = {}
    for block in assembled_blocks:
        for local_p, global_p in enumerate(block.point_indices.tolist()):
            if global_p not in point_to_xyz:
                point_to_xyz[global_p] = block.shape[:, local_p]   # (3,)

    global_scales  = torch.zeros(len(frame_indices), device=device, dtype=dtype)
    global_offsets = torch.zeros(len(frame_indices), device=device, dtype=dtype)

    for fi, (f, cam) in enumerate(zip(frame_indices, cam_lists)):
        # Points observed in this frame
        lam_row = lambda_mat[f]                      # (P_orig,)
        obs_mask = torch.isfinite(lam_row)           # (P_orig,)
        obs_ids  = obs_mask.nonzero(as_tuple=True)[0]

        # Keep only those with a global 3D position
        valid_ids = [p.item() for p in obs_ids if p.item() in point_to_xyz]
        if len(valid_ids) < 4:
            # Fall back to identity calibration
            global_scales[fi]  = 1.0
            global_offsets[fi] = 0.0
            continue

        # Stack 3D points and compute projective depths under global camera
        xyz = torch.stack([point_to_xyz[p] for p in valid_ids], dim=1)   # (3, K)
        xyz_hom = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=device, dtype=dtype)], dim=0)  # (4, K)
        proj    = cam.to(dtype=dtype) @ xyz_hom          # (3, K)
        lam_global = proj[2]                              # (K,) projective depths

        # Raw MDE depths at these point locations
        lam_mde = lam_row[torch.tensor(valid_ids, device=device)]   # (K,)

        # Only use points with positive projective depth
        pos_mask = lam_global > 0
        if pos_mask.sum() < 4:
            global_scales[fi]  = 1.0
            global_offsets[fi] = 0.0
            continue

        s, o = fit_affine(lam_mde[pos_mask], lam_global[pos_mask])

        lam_pred = s * lam_mde[pos_mask] + o
        residuals = lam_pred - lam_global[pos_mask]
        print(f"s={s:.4f}, o={o:.4f}")
        print(f"fit residual mean={residuals.mean():.4f}, std={residuals.std():.4f}")

        global_scales[fi]  = s
        global_offsets[fi] = o

    return global_scales, global_offsets


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_block_reconstruction(
    W_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    block_size: int = 4,
    overlap: int = 2,
    min_common: int = 6,
    run_kwargs: dict = None,
) -> dict:
    """
    Full block reconstruction pipeline.

    Args:
        W_mat        : (3F, P) observation matrix, NaN where missing
        lambda_mat   : (F, P) depth matrix, NaN where missing
        block_size   : number of frames per block
        overlap      : number of overlap frames between consecutive blocks
        min_common   : minimum common points needed to stitch two blocks
        run_kwargs   : extra kwargs forwarded to run_projective_reconstruction

    Returns dict with:
        cam_lists      : list of (3,4) camera matrices, one per surviving global frame
        frame_indices  : list of global frame indices corresponding to cam_lists
        shape_dict     : dict {global_point_idx: (3,) tensor}
        global_scales  : (F_surv,) per-frame depth scale  — apply to raw MDE before backprojection
        global_offsets : (F_surv,) per-frame depth offset — apply to raw MDE before backprojection
        H_chain        : list of (4,4) stitching homographies
        consistency    : list of per-boundary consistency errors
        block_results  : list of raw BlockResult before stitching
        assembled_blocks: list of BlockResult after stitching
    """
    if run_kwargs is None:
        run_kwargs = {}

    F = lambda_mat.shape[0]
    P = lambda_mat.shape[1]
    device = W_mat.device

    global_point_mask = torch.ones(P, dtype=torch.bool, device=device)

    # Stage 1: partition
    frame_blocks = make_blocks(F, block_size=block_size, overlap=overlap)
    print(f"Partitioned {F} frames into {len(frame_blocks)} blocks (size={block_size}, overlap={overlap})")
    
    # Stage 2: local reconstruction
    block_results = []
    for i, frame_ids in enumerate(frame_blocks):
        print(f"  Block {i+1}/{len(frame_blocks)}: frames {frame_ids}")
        br = reconstruct_block(W_mat, lambda_mat, frame_ids, global_point_mask, run_kwargs)
        if br is not None:
            block_results.append(br)
            print(f"    -> {len(br.frame_indices)} frames, {br.point_indices.shape[0]} points")
        else:
            print(f"    -> FAILED, skipping")

    if not block_results:
        raise RuntimeError("All blocks failed reconstruction.")

    # Stage 3+4: stitch and assemble
    result = assemble_blocks(block_results, min_common=min_common)
    result["block_results"] = block_results

    # Stage 5: fit global per-frame depth calibration
    global_scales, global_offsets = compute_global_depth_calibration(
        assembled_blocks=result["assembled_blocks"],
        lambda_mat=lambda_mat,
        cam_lists=result["cam_lists"],
        frame_indices=result["frame_indices"],
    )
    result["global_scales"]  = global_scales
    result["global_offsets"] = global_offsets

    print(f"\nAssembly complete: {len(result['cam_lists'])} cameras, "
          f"{len(result['shape_dict'])} points")

    return result


def assemble_blocks(blocks: list[BlockResult], min_common: int = 6) -> dict:
    """
    Chain-stitch all blocks into the projective frame of blocks[0].
    """
    if not blocks:
        return {}

    assembled = [blocks[0]]
    H_chain     = []
    consistency = []

    for i in range(1, len(blocks)):
        prev = assembled[-1]
        curr = blocks[i]

        H = estimate_H(prev, curr, min_points=min_common)
        if H is None:
            print(f"  Stitching failed at block {i}, skipping block.")
            continue

        err = consistency_error(prev, curr, H)
        consistency.append(err)
        print(f"  Block {i}: consistency error = {err:.4f}, "
              f"common points = {find_common_points(prev, curr)[0].numel()}")

        assembled.append(apply_H_to_block(curr, H))
        H_chain.append(H)

    # Cameras: first-block estimate wins for each global frame
    frame_to_cam = {}
    for block in assembled:
        for local_f, global_f in enumerate(block.frame_indices):
            if global_f not in frame_to_cam:
                frame_to_cam[global_f] = block.motion[local_f * 3: local_f * 3 + 3]

    sorted_frames = sorted(frame_to_cam.keys())
    cam_lists = [frame_to_cam[f] for f in sorted_frames]

    # Shape: average over blocks for points seen in multiple
    point_accum = {}
    for block in assembled:
        for local_p, global_p in enumerate(block.point_indices.tolist()):
            pt = block.shape[:, local_p]
            if global_p not in point_accum:
                point_accum[global_p] = []
            point_accum[global_p].append(pt)

    shape_dict = {k: torch.stack(v).mean(0) for k, v in point_accum.items()}

    return dict(
        cam_lists=cam_lists,
        frame_indices=sorted_frames,
        shape_dict=shape_dict,
        H_chain=H_chain,
        consistency=consistency,
        assembled_blocks=assembled,
    )