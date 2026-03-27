import math
import matplotlib.pyplot as plt
import numpy as np
import k3d
import torch

import numpy as np
import torch
import k3d

def get_orthogonal_camera_vectors(P):
    """
    Decomposes P = K[R|t] to find the true orthogonal rotation R.
    """
    if torch.is_tensor(P):
        P = P.detach().cpu().numpy()
    
    # 1. Handle 2x4 (Affine) vs 3x4 (Projective)
    if P.shape[0] == 2:
        M = np.vstack([P[:, :3], [0, 0, 0]]) # Pad to 3x3
        t = np.vstack([P[:, 3:], [1]])      # Pad to 3x1
    else:
        M = P[:3, :3]
        t = P[:3, 3:]

    # 2. Extract Camera Center
    try:
        C = (-np.linalg.inv(M) @ t).flatten()
    except np.linalg.LinAlgError:
        C = np.array([0,0,0])

    # 3. RQ Decomposition to isolate pure Rotation (R) from Intrinsics (K)
    # Using the flip-QR-flip trick
    M_flipped = np.flipud(M).T
    Q_q, R_r = np.linalg.qr(M_flipped)
    
    R_ortho = np.flipud(Q_q.T)
    
    # Ensure a right-handed system (det == 1)
    if np.linalg.det(R_ortho) < 0:
        R_ortho *= 1

    # 4. Extract Orthogonal Axes
    # Row 0 = Right, Row 1 = Up (inverted for screen space), Row 2 = Forward
    right = R_ortho[0, :]
    up = -R_ortho[1, :] 
    forward = R_ortho[2, :]
    
    return C, right, up, forward

def k3d_3d_plot(point_input, color_input=None, camera_input=None, scale=70):
    plot = k3d.plot(camera_auto_fit=True)

    default_colors = [
        0xff0000, 0x00ff00, 0x0000ff, 0xffff00,
        0x00ffff, 0xff00ff, 0xffa500, 0x800080
    ]

    # -------------------------
    # POINTS
    # -------------------------
    if point_input is not None:
        if not isinstance(point_input, (list, tuple)):
            point_input = [point_input]
        if color_input is not None and not isinstance(color_input, (list, tuple)):
            color_input = [color_input]

        pts_list = []
        for p in point_input:
            if torch.is_tensor(p): p = p.detach().cpu().numpy()
            p = p.T if p.shape[0] == 3 else p
            pts_list.append(p.astype(np.float32))

        combined = np.vstack(pts_list)
        mins, maxs = combined.min(axis=0), combined.max(axis=0)
        global_extent = float(np.linalg.norm(maxs - mins)) or 1.0
        psize = global_extent / scale

        for i, p in enumerate(pts_list):
            params = {"point_size": psize, "name": f"Point Set {i}"}
            if color_input is not None and i < len(color_input):
                c = color_input[i]
                if torch.is_tensor(c): c = c.detach().cpu().numpy()
                if c.ndim == 2 and c.shape[1] == 3:
                    if c.max() <= 1.1: c = (c * 255)
                    c = c.astype(np.uint32)
                    c = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
                params["colors"] = c.astype(np.uint32)
            else:
                params["color"] = default_colors[i % len(default_colors)]
            plot += k3d.points(p, **params)
    else:
        global_extent = 1.0

    # -------------------------
    # CAMERAS — fully vectorized
    # -------------------------
    if camera_input is not None:
        if not isinstance(camera_input, (list, tuple)):
            camera_input = [camera_input]

        N = len(camera_input)
        cam_size = global_extent * 0.12
        dist = cam_size
        w, h = cam_size * 0.8, cam_size * 0.6
        a_len = cam_size * 0.4

        # Batch-extract all camera vectors: (N, 3) each
        results = np.array([get_orthogonal_camera_vectors(P) for P in camera_input], dtype=object)
        C = np.stack(results[:, 0]).astype(np.float32)  # (N, 3)
        R = np.stack(results[:, 1]).astype(np.float32)
        U = np.stack(results[:, 2]).astype(np.float32)
        F = np.stack(results[:, 3]).astype(np.float32)

        # Frustum corners: (N, 3) each
        Fd = F * dist
        c1 = C + Fd + R*w + U*h
        c2 = C + Fd - R*w + U*h
        c3 = C + Fd - R*w - U*h
        c4 = C + Fd + R*w - U*h

        # Tripod endpoints: (N, 3) each
        Ra = C + R * a_len
        Ua = C + U * a_len
        Fa = C + F * a_len

        # Per-camera vertices: 11 verts — C,c1,c2,c3,c4, C,Ra, C,Ua, C,Fa
        # Stack into (N, 11, 3), then flatten to (N*11, 3)
        verts = np.stack([C, c1, c2, c3, c4, C, Ra, C, Ua, C, Fa], axis=1)  # (N, 11, 3)
        all_verts = verts.reshape(-1, 3)  # (N*11, 3)

        # Index template for one camera (11 verts, 0-indexed)
        idx_template = np.array([
            [0,1],[0,2],[0,3],[0,4],   # frustum
            [1,2],[2,3],[3,4],[4,1],   # frustum rect
            [5,6],                     # R axis
            [7,8],                     # U axis
            [9,10]                     # F axis
        ], dtype=np.uint32)            # (11, 2)

        # Broadcast offsets: each camera block starts at i*11
        offsets = (np.arange(N) * 11).reshape(N, 1, 1).astype(np.uint32)  # (N,1,1)
        all_inds = (idx_template[None] + offsets).reshape(-1, 2)           # (N*11, 2)

        plot += k3d.lines(
            all_verts,
            all_inds,
            color=0xff0000,
            width=float(cam_size * 0.02),
            shader="simple"
        )

        plot += k3d.points(
            C,
            point_size=float(cam_size * 0.12),
            color=0x000000,
            shader="flat"
        )

    plot.display()


def rerun_3d_plot(point_input, color_input=None, camera_input=None, entity_prefix="scene"):
    """
    Rerun-based alternative to k3d_3d_plot. Much faster for large point clouds.
    Call rr.init(..., spawn=True) once in your notebook before using this.

    Args:
        point_input    : (3, N) or (N, 3) tensor/array, or list of such arrays.
        color_input    : (N, 3) float [0,1] or uint8 RGB array, or list of such arrays.
        camera_input   : list of (3, 4) camera matrices [R | t].
        entity_prefix  : rerun entity path prefix (change to avoid overwriting previous logs).
    """
    import rerun as rr

    # ---- Points ----
    if point_input is not None:
        if not isinstance(point_input, (list, tuple)):
            point_input = [point_input]
        if color_input is not None and not isinstance(color_input, (list, tuple)):
            color_input = [color_input]

        for i, p in enumerate(point_input):
            if torch.is_tensor(p): p = p.detach().cpu().numpy()
            p = p.T if p.shape[0] == 3 else p          # ensure (N, 3)
            p = p.astype(np.float32)

            kwargs = {}
            if color_input is not None and i < len(color_input):
                c = color_input[i]
                if torch.is_tensor(c): c = c.detach().cpu().numpy()
                if c.max() <= 1.01: c = (c * 255).astype(np.uint8)
                kwargs["colors"] = c.astype(np.uint8)

            rr.log(f"{entity_prefix}/points/{i}", rr.Points3D(p, **kwargs))

    # ---- Cameras ----
    if camera_input is not None:
        if not isinstance(camera_input, (list, tuple)):
            camera_input = [camera_input]

        for i, P in enumerate(camera_input):
            if torch.is_tensor(P): P = P.detach().cpu().numpy()
            R = P[:3, :3]
            t = P[:3,  3]
            C = -R.T @ t  # camera center in world coords

            # Rerun expects world-from-camera transform
            rr.log(
                f"{entity_prefix}/cameras/{i}",
                rr.Transform3D(
                    translation=C.astype(np.float32),
                    mat3x3=R.T.astype(np.float32),
                ),
            )
            rr.log(
                f"{entity_prefix}/cameras/{i}",
                rr.Pinhole(focal_length=100.0, width=160, height=120),
            )


def plot_depth_tensor_grid(
    depth_tensor,  # (num_frames, H, W, 1) or (num_frames, H, W)
    nrows=None,
    ncols=None,
    figsize=(15, 10),
    start_frame=0,
    obs_mat=None,
    features=None,
    show=True,
):
    """
    Display a grid of depth frames with optional 2D feature scatter overlay.

    Args:
        depth_tensor: torch tensor or numpy array of shape (F, H, W, 1) or (F, H, W)
        nrows, ncols: optional; if not provided, computed automatically
        figsize: tuple, figure size
        start_frame: first frame index to display
        obs_mat: optional observation matrix [2F, P]
        features: optional indices for tracked features
        show: whether to call plt.show()

    Returns:
        (fig, axes)
    """
    # Convert to numpy if torch tensor
    if hasattr(depth_tensor, "detach"):
        depth = depth_tensor.detach().cpu().squeeze().numpy()
    else:
        depth = depth_tensor.squeeze()

    total_frames = depth.shape[0]
    if total_frames == 0:
        raise ValueError("Depth tensor has zero frames.")

    # Compute number of frames to plot
    frames_to_plot = max(0, total_frames - start_frame)
    end_frame = min(start_frame + frames_to_plot, total_frames)

    # Auto grid size if not provided
    if nrows is None or ncols is None:
        ncols = int(math.ceil(math.sqrt(frames_to_plot)))
        nrows = int(math.ceil(frames_to_plot / ncols))

    # Precompute shared color range to avoid per-frame normalization
    depth_slice = depth[start_frame:end_frame]
    vmin, vmax = float(depth_slice.min()), float(depth_slice.max())
    frame_maxes = depth_slice.min(axis=(1, 2))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) or axes.ndim > 0 else [axes]

    total_slots = len(axes)
    has_overlay = obs_mat is not None and features is not None

    for i in range(frames_to_plot):
        idx = start_frame + i
        if idx >= total_frames:
            break

        ax = axes[i]
        ax.imshow(depth[idx], vmin=vmin, vmax=vmax, interpolation='nearest', cmap='plasma_r')
        if has_overlay:
            ax.plot(
                obs_mat[idx * 2, features],
                obs_mat[idx * 2 + 1, features],
                'r.', ms=3,
            )
        ax.set_title(f"Frame {idx + 1}, min depth: {frame_maxes[i]:.2f}")
        ax.axis("off")

    # Hide unused subplots
    for j in range(frames_to_plot, total_slots):
        axes[j].axis("off")

    fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
