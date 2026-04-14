"""
Pairwise relative pose estimation using MADPose.

Integrates with the existing pipeline:
    obs_mat      [2F, N]  tracked keypoints (u,v interleaved per frame)
    lambda_mat   [F, N]   pre-sampled depths at keypoints
    Ks           [F,3,3]  per-frame intrinsics

References: https://github.com/MarkYu98/madpose
"""

import sys
sys.path.append("../madpose/")

import torch
import numpy as np
import madpose


def make_madpose_options(
    min_iters: int = 100,
    max_iters: int = 1000,
    threshold_px: float = 4.0,
    epipolar_threshold: float = 1.0,
    num_lo_steps: int = 4,
    random_seed: int = 0,
) -> madpose.HybridLORansacOptions:
    options = madpose.HybridLORansacOptions()
    options.min_num_iterations = min_iters
    options.max_num_iterations = max_iters
    options.success_probability = 0.9999
    options.random_seed = random_seed
    options.final_least_squares = True
    options.threshold_multiplier = 5.0
    options.num_lo_steps = num_lo_steps
    options.squared_inlier_thresholds = [threshold_px ** 2, epipolar_threshold ** 2]
    options.data_type_weights = [1.0, 1.0]
    return options


def make_madpose_config(
    min_depth_constraint: bool = True,
    use_shift: bool = True,
    num_threads: int = 8,
) -> madpose.EstimatorConfig:
    cfg = madpose.EstimatorConfig()
    cfg.min_depth_constraint = min_depth_constraint
    cfg.use_shift = use_shift
    cfg.ceres_num_threads = num_threads
    return cfg


def estimate_pairwise_pose(
    obs_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    Ks: torch.Tensor,
    frame_i: int,
    frame_j: int,
    options: madpose.HybridLORansacOptions | None = None,
    est_config: madpose.EstimatorConfig | None = None,
) -> dict:
    """
    Estimate relative pose between two frames using MADPose.

    Args:
        obs_mat:     [2F, N] tracked keypoints (u, v interleaved per frame)
        lambda_mat:  [F, N]  pre-sampled depths at keypoints (e.g. tracks_lambda_*)
        Ks:          [F, 3, 3] or [1, 3, 3] camera intrinsics
        frame_i:     index of the first frame
        frame_j:     index of the second frame

    Returns:
        dict with keys: R, t, scale, offset0, offset1, num_inliers, inlier_ratio, pose, stats
    """
    if options is None:
        options = make_madpose_options()
    if est_config is None:
        est_config = make_madpose_config()

    depth_i = lambda_mat[frame_i]  # [N]
    depth_j = lambda_mat[frame_j]  # [N]

    mkpts_i = obs_mat[frame_i * 2: frame_i * 2 + 2, :].T  # [N, 2]
    mkpts_j = obs_mat[frame_j * 2: frame_j * 2 + 2, :].T  # [N, 2]

    # filter points with invalid depths in either frame
    valid = (depth_i > 0) & (depth_j > 0) & torch.isfinite(depth_i) & torch.isfinite(depth_j)
    depth_i = depth_i[valid]
    depth_j = depth_j[valid]
    mkpts_i = mkpts_i[valid]
    mkpts_j = mkpts_j[valid]

    if valid.sum() < 10:
        raise ValueError(f"Too few valid depth points after filtering: {valid.sum()} (frames {frame_i},{frame_j})")

    K_i = Ks[min(frame_i, Ks.shape[0] - 1)]
    K_j = Ks[min(frame_j, Ks.shape[0] - 1)]

    pose, stats = madpose.HybridEstimatePoseScaleOffset(
        mkpts_i.cpu().numpy().astype(np.float64),
        mkpts_j.cpu().numpy().astype(np.float64),
        depth_i.cpu().numpy().astype(np.float64),
        depth_j.cpu().numpy().astype(np.float64),
        [float(depth_i.min()), float(depth_j.min())],
        K_i.cpu().numpy().astype(np.float64),
        K_j.cpu().numpy().astype(np.float64),
        options,
        est_config,
    )

    return {
        "R": torch.tensor(pose.R(), dtype=torch.float64),
        "t": torch.tensor(pose.t(), dtype=torch.float64),
        "scale": pose.scale,
        "offset0": pose.offset0,
        "offset1": pose.offset1,
        "num_inliers": stats.best_num_inliers,
        "inlier_ratio": stats.best_num_inliers / int(valid.sum()),
        "pose": pose,
        "stats": stats,
    }


def estimate_all_pairs(
    obs_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    Ks: torch.Tensor,
    pairs: list[tuple[int, int]],
    options: madpose.HybridLORansacOptions | None = None,
    est_config: madpose.EstimatorConfig | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run pairwise madpose estimation for a list of (i, j) frame pairs.

    Args:
        obs_mat:    [2F, N]
        lambda_mat: [F, N]  pre-sampled depths at keypoints
        Ks:         [F, 3, 3]
        pairs:      list of (frame_i, frame_j) tuples
    """
    if options is None:
        options = make_madpose_options()
    if est_config is None:
        est_config = make_madpose_config()

    results = []
    iterable = pairs
    if verbose:
        from tqdm import tqdm
        iterable = tqdm(pairs, desc="MADPose pairwise estimation")

    for (i, j) in iterable:
        result = estimate_pairwise_pose(obs_mat, lambda_mat, Ks, i, j, options, est_config)
        result["frame_i"] = i
        result["frame_j"] = j
        results.append(result)

        if verbose:
            iterable.set_postfix({
                f"({i},{j})": f"inliers={result['num_inliers']} "
                              f"ratio={result['inlier_ratio']:.2f} "
                              f"scale={result['scale']:.3f}"
            })

    return results


def build_pose_graph(results: list[dict]) -> dict:
    """
    Convert pairwise results into an edge dict keyed by (i, j).
    """
    return {
        (r["frame_i"], r["frame_j"]): {
            "R": r["R"],
            "t": r["t"],
            "scale": r["scale"],
            "offset0": r["offset0"],
            "offset1": r["offset1"],
            "inlier_ratio": r["inlier_ratio"],
            "num_inliers": r["num_inliers"],
        }
        for r in results
    }
