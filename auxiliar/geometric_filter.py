"""
geometric_filter.py
-------------------
Geometric filtering of a multi-frame observation matrix.

For each pair of frames, fit a fundamental matrix (or homography) with
RANSAC and count how many pairs each point is an inlier in.  Points that
are inliers in fewer than min_vote_fraction of their testable pairs are
discarded.

Only wide-baseline pairs are tested (min_frame_gap) because consecutive
frames give near-degenerate F matrices where almost everything passes.
"""

import numpy as np
import torch
from itertools import combinations
from typing import Optional, Literal


def _norm_pts(pts):
    """Hartley normalisation.  pts: (N,2) → (N,3), T (3,3)"""
    mu = pts.mean(0)
    sc = np.sqrt(2) / (np.std(pts) + 1e-12)
    T  = np.array([[sc,  0, -sc*mu[0]],
                   [ 0, sc, -sc*mu[1]],
                   [ 0,  0,         1]], dtype=np.float64)
    return (T @ np.column_stack([pts, np.ones(len(pts))]).T).T, T


def _fit_F(p1, p2):
    """Normalised 8-point fundamental matrix.  p1,p2: (N,2)"""
    p1n, T1 = _norm_pts(p1)
    p2n, T2 = _norm_pts(p2)
    x1, y1 = p1n[:,0], p1n[:,1]
    x2, y2 = p2n[:,0], p2n[:,1]
    A = np.column_stack([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(len(p1))])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0
    F = T2.T @ (U @ np.diag(S) @ Vt2) @ T1
    return F / (np.linalg.norm(F) + 1e-15)


def _sampson(F, p1, p2):
    """Sampson distance.  Returns (N,) errors."""
    x1 = np.column_stack([p1, np.ones(len(p1))]).T   # (3,N)
    x2 = np.column_stack([p2, np.ones(len(p2))]).T
    Fx1  = F  @ x1
    Ftx2 = F.T @ x2
    num  = (x2 * Fx1).sum(0)**2
    den  = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-15
    return num / den


def _fit_H(p1, p2):
    """DLT homography.  p1,p2: (N,2)"""
    N = len(p1)
    x,  y  = p1[:,0], p1[:,1]
    xp, yp = p2[:,0], p2[:,1]
    z = np.zeros(N); o = np.ones(N)
    r0 = np.column_stack([-x, -y, -o,  z,  z,  z,  xp*x,  xp*y,  xp])
    r1 = np.column_stack([ z,  z,  z, -x, -y, -o,  yp*x,  yp*y,  yp])
    A  = np.vstack([r0, r1])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    H = Vt[-1].reshape(3, 3)
    return H / (H[2,2] + 1e-15)


def _sym_transfer(H, p1, p2):
    """Symmetric transfer error for homography.  Returns (N,)."""
    p1h = np.column_stack([p1, np.ones(len(p1))]).T
    p2h = np.column_stack([p2, np.ones(len(p2))]).T
    q2  = H @ p1h;  q2 /= q2[2:3] + 1e-15
    d2  = ((q2[:2] - p2h[:2])**2).sum(0)
    Hi  = np.linalg.inv(H + np.eye(3)*1e-10)
    q1  = Hi @ p2h; q1 /= q1[2:3] + 1e-15
    d1  = ((q1[:2] - p1h[:2])**2).sum(0)
    return d1 + d2


def _ransac(p1, p2, mode, thresh, max_iters=500):
    """Simple RANSAC loop.  Returns (N,) bool inlier mask."""
    N       = len(p1)
    min_pts = 4 if mode == 'H' else 8
    if N < min_pts:
        return np.zeros(N, dtype=bool)

    best_inl = np.zeros(N, dtype=bool)
    best_cnt = 0

    for _ in range(max_iters):
        idx = np.random.choice(N, min_pts, replace=False)
        try:
            if mode == 'H':
                M   = _fit_H(p1[idx], p2[idx])
                err = _sym_transfer(M, p1, p2)
            else:
                M   = _fit_F(p1[idx], p2[idx])
                err = _sampson(M, p1, p2)
        except np.linalg.LinAlgError:
            continue

        inl = err < thresh**2
        cnt = inl.sum()
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl

    # refit on inliers
    if best_cnt >= min_pts:
        try:
            if mode == 'H':
                M   = _fit_H(p1[best_inl], p2[best_inl])
                err = _sym_transfer(M, p1, p2)
            else:
                M   = _fit_F(p1[best_inl], p2[best_inl])
                err = _sampson(M, p1, p2)
            best_inl = err < thresh**2
        except np.linalg.LinAlgError:
            pass

    return best_inl


def geometric_filter_obs(
    obs_mat: torch.Tensor,
    K: Optional[torch.Tensor] = None,
    mode: Literal['H', 'F', 'E'] = 'F',
    min_vote_fraction: float = 0.4,
    min_inlier_pairs: int = 1,
    ransac_thresh: float = 2.0,
    min_pair_points: int = 8,
    min_frame_gap: int = 3,
    max_pairs: Optional[int] = 300,
    max_ransac_iters: int = 500,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    obs_mat           (2F, P)  pixel coords, NaN where missing
    lambda_mat        (F,  P)  depths, NaN where missing
    K                 (3,3) intrinsics — ignored for F, used to convert F→E
    mode              'H' homography | 'F' fundamental | 'E' essential
    min_vote_fraction point must be inlier in this fraction of testable pairs
    min_inlier_pairs  absolute minimum inlier pairs (floor on the above)
    ransac_thresh     reprojection / Sampson threshold in pixels
    min_frame_gap     skip pairs with |fi-fj| < this (avoids degenerate F)
    min_pair_points   skip pair if fewer co-visible points
    max_pairs         cap on pairs evaluated (None = all)
    max_ransac_iters  RANSAC iterations per pair

    Returns
    -------
    obs_clean    (2F, P_kept)
    lam_clean    (F,  P_kept)
    inlier_mask  (P,) bool tensor
    """
    F  = obs_mat.shape[0] // 2
    P  = obs_mat.shape[1]

    K_np = None
    if K is not None and mode == 'E':
        K_np = K.float().cpu().numpy()
        while K_np.ndim > 2:
            K_np = K_np.squeeze(0)

    obs_np = obs_mat.float().cpu().numpy()
    xs_np  = obs_np[0::2]   # (F, P)

    # select pairs
    all_pairs = [(i, j) for i, j in combinations(range(F), 2)
                 if j - i >= min_frame_gap]
    if max_pairs is not None and len(all_pairs) > max_pairs:
        rng       = np.random.default_rng(42)
        all_pairs = [all_pairs[k] for k in
                     rng.choice(len(all_pairs), max_pairs, replace=False).tolist()]

    if verbose:
        print(f"Geometric filter | mode={mode}  F={F}  P={P}  "
              f"pairs={len(all_pairs)}  gap>={min_frame_gap}  thresh={ransac_thresh}px")

    inlier_votes = np.zeros(P, dtype=np.int32)
    testable_cnt = np.zeros(P, dtype=np.int32)

    for k, (fi, fj) in enumerate(all_pairs):
        # co-visible points for this pair
        valid = (~np.isnan(xs_np[fi])) & (~np.isnan(xs_np[fj]))
        vi    = np.where(valid)[0]
        if len(vi) < min_pair_points:
            continue

        p1 = np.stack([obs_np[2*fi,   vi], obs_np[2*fi+1, vi]], 1).astype(np.float64)
        p2 = np.stack([obs_np[2*fj,   vi], obs_np[2*fj+1, vi]], 1).astype(np.float64)

        # drop non-finite rows before RANSAC
        ok   = np.isfinite(p1).all(1) & np.isfinite(p2).all(1)
        vi_ok = vi[ok]
        if ok.sum() < min_pair_points:
            continue

        inl = _ransac(p1[ok], p2[ok], mode, ransac_thresh, max_ransac_iters)

        testable_cnt[vi_ok]      += 1
        inlier_votes[vi_ok[inl]] += 1

        if verbose and (k+1) % max(1, len(all_pairs)//10) == 0:
            pct = (k+1) / len(all_pairs) * 100
            print(f"  {pct:4.0f}%  pair ({fi:2d},{fj:2d})  "
                  f"co-vis={ok.sum():4d}  inliers={inl.sum():4d}  "
                  f"({100*inl.mean():.0f}%)")

    # survival rule: inlier in >= fraction of testable pairs
    min_votes    = np.where(testable_cnt > 0,
                            np.maximum(np.ceil(testable_cnt * min_vote_fraction).astype(int),
                                       min_inlier_pairs),
                            0)
    inlier_mask  = inlier_votes >= min_votes
    # never-tested points (short tracks that never formed a wide pair): keep
    inlier_mask |= testable_cnt == 0

    n_kept = int(inlier_mask.sum())
    if verbose:
        tested = testable_cnt > 0
        print(f"  Kept {n_kept}/{P} ({100*n_kept/P:.1f}%)  |  "
              f"among tested: {100*inlier_mask[tested].mean():.1f}%")

    mask_t    = torch.from_numpy(inlier_mask)
    return obs_mat[:, mask_t], mask_t


def build_filtered_inputs(
    obs_mat: torch.Tensor,
    lambda_mat: torch.Tensor,
    K: Optional[torch.Tensor] = None,
    mode: Literal['H', 'F', 'E'] = 'F',
    min_vote_fraction: float = 0.4,
    min_inlier_pairs: int = 1,
    ransac_thresh: float = 2.0,
    min_pair_points: int = 8,
    min_frame_gap: int = 3,
    max_pairs: Optional[int] = 300,
    max_ransac_iters: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Convenience wrapper: filter + build homogeneous W_mat.

    Returns dict with keys:
        W_mat        (3F, P_kept)
        lambda_mat   (F,  P_kept)
        inlier_mask  (P_orig,) bool
        obs_clean    (2F, P_kept)
    """
    obs_c, lam_c, mask = geometric_filter_obs(
        obs_mat, lambda_mat,
        K=K, mode=mode,
        min_vote_fraction=min_vote_fraction,
        min_inlier_pairs=min_inlier_pairs,
        ransac_thresh=ransac_thresh,
        min_pair_points=min_pair_points,
        min_frame_gap=min_frame_gap,
        max_pairs=max_pairs,
        max_ransac_iters=max_ransac_iters,
        verbose=verbose,
    )

    F  = obs_c.shape[0] // 2
    P  = obs_c.shape[1]
    x  = obs_c[0::2]
    y  = obs_c[1::2]
    W  = torch.stack([x, y, torch.ones_like(x)], 1).reshape(3*F, P)
    W[torch.isnan(x).repeat_interleave(3, 0)] = float('nan')

    return dict(W_mat=W, lambda_mat=lam_c, inlier_mask=mask, obs_clean=obs_c)