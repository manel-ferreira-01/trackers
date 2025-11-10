import torch
import torch.nn.functional as F

def sample_depths(depths, tracks):
    """
    depths: [F, H, W]  depth maps
    tracks: [2F, P]    pixel coords (u,v)
    Returns: [F, P] sampled depths
    """
    F_, H, W = depths.shape
    P = tracks.shape[1]

    depths = depths.unsqueeze(1)  # [F, 1, H, W]

    all_depths = []
    for f in range(F_):
        u = tracks[2*f, :]   # [P]
        v = tracks[2*f+1, :] # [P]

        # Normalize coords to [-1,1] for grid_sample
        u_norm = (u / (W-1)) * 2 - 1
        v_norm = (v / (H-1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, P, 1, 2)  # [1,P,1,2]

        z = F.grid_sample(
            depths[f:f+1], grid, align_corners=True, mode='bilinear'
        ).view(-1)  # [P]

        all_depths.append(z)

    return torch.stack(all_depths, dim=0)  # [F, P]


def build_depth_weighted_matrix(tracks, depths, fx, fy, cx, cy):
    """
    Build a 3F x P depth-weighted observation matrix.

    Args:
        tracks: torch.Tensor [2F, P]
            Pixel coordinates (u,v) for each feature across frames.
        depths: torch.Tensor [F, H, W]
            Depth map for each frame.
        fx, fy, cx, cy: floats
            Camera intrinsics (assumed same for all frames).

    Returns:
        W_proj: torch.Tensor [3F, P]
            Depth-weighted homogeneous observation matrix.
    """
    F = depths.shape[0]
    P = tracks.shape[1]

    # interpolator of depth at 
    z = sample_depths(depths, tracks) 

    u = tracks[0::2, :]  # [F, P]
    v = tracks[1::2, :]  # [F, P]

    # Normalize to camera plane
    x_norm = (u - cx) / fx  # [F, P]
    y_norm = (v - cy) / fy  # [F, P]

    W_proj = torch.zeros((3 * F, P), dtype=tracks.dtype, device=tracks.device)

    # Fill each frame’s block of 3 rows
    for f in range(F):
        W_proj[3 * f + 0, :] = x_norm[f] * z[f]
        W_proj[3 * f + 1, :] = y_norm[f] * z[f]
        W_proj[3 * f + 2, :] = z[f]

    return W_proj, z


def normalize_measurement_matrix(W):
    """
    Isotropic normalization for 3F x P projective measurement matrix.

    Args:
        W: torch.Tensor [3F, P]
            Depth-weighted homogeneous measurement matrix.

    Returns:
        W_norm: torch.Tensor [3F, P] normalized matrix
        T_list: list of torch.Tensor [4,4]
            Normalization transforms per frame (for upgrading back).
    """
    F = W.shape[0] // 3
    P = W.shape[1]

    W_norm = torch.zeros_like(W)
    T_list = []

    for f in range(F):
        block = W[3*f:3*f+3, :]  # [3, P]

        homog = torch.cat([block, torch.ones(1, P, dtype=W.dtype, device=W.device)], dim=0)

        # Dehomogenize
        X = homog[0, :] / homog[3, :]
        Y = homog[1, :] / homog[3, :]
        Z = homog[2, :] / homog[3, :]

        # Centroid
        mean_x, mean_y, mean_z = X.mean(), Y.mean(), Z.mean()

        # Shift
        Xs = X - mean_x
        Ys = Y - mean_y
        Zs = Z - mean_z

        mean_dist = torch.mean(torch.sqrt(Xs**2 + Ys**2 + Zs**2))

        s = torch.sqrt(torch.tensor(3.0, dtype=W.dtype, device=W.device)) / mean_dist
        #print(3)

        T = torch.tensor([
            [s, 0, 0, -s*mean_x],
            [0, s, 0, -s*mean_y],
            [0, 0, s, -s*mean_z],
            [0, 0, 0, 1]
        ], dtype=W.dtype, device=W.device)

        homog_norm = T @ homog
        W_norm[3*f:3*f+3, :] = homog_norm[:3, :]

        T_list.append(T)

    return W_norm, T_list
