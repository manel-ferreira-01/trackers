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
    """
    Visualizes 3D points and cameras using K3D.
    
    Args:
        point_input: 
            - A single Nx3 array/tensor of points.
            - OR a list of [N1x3, N2x3, ...] arrays/tensors.
        color_input: (Optional)
            - If None: Each set in point_input gets a unique color from a default palette.
            - If provided: Must match the structure of point_input.
            - Can be a list of [N1x3, N2x3, ...] RGB values (0-1 float or 0-255 int).
            - Can be a list of [N1, N2, ...] packed uint32 integers (0xRRGGBB).
        camera_input: (Optional)
            - A list of 3x4 or 4x4 camera projection matrices (World-to-Camera or Cam-to-World).
        scale: 
            - Float used to determine point and camera size relative to scene extent.
    """
    
    plot = k3d.plot(camera_auto_fit=True)

    # Standard colormap for when no color_input is provided
    default_colors = [
        0xff0000, 0x00ff00, 0x0000ff, 0xffff00,
        0x00ffff, 0xff00ff, 0xffa500, 0x800080
    ]

    # -------------------------
    # POINTS
    # -------------------------
    if point_input is not None:
        # Normalize inputs to lists
        if not isinstance(point_input, (list, tuple)):
            point_input = [point_input]
        
        if color_input is not None and not isinstance(color_input, (list, tuple)):
            color_input = [color_input]

        pts_list = []
        for p in point_input:
            if torch.is_tensor(p): p = p.detach().cpu().numpy()
            p = p.T if p.shape[0] == 3 else p
            pts_list.append(p.astype(np.float32))

        # Calculate scale based on the whole scene
        combined = np.vstack(pts_list)
        mins, maxs = combined.min(axis=0), combined.max(axis=0)
        global_extent = np.linalg.norm(maxs - mins)
        if global_extent == 0: global_extent = 1.0
        psize = float(global_extent / scale)

        # Plot each set
        for i, p in enumerate(pts_list):
            params = {
                "point_size": psize,
                "name": f"Point Set {i}",
                #"shader": "flat"
            }

            if color_input is not None and i < len(color_input):
                # PER-POINT COLOR MODE
                c = color_input[i]
                if torch.is_tensor(c): c = c.detach().cpu().numpy()
                
                # If colors are [N, 3], pack them into uint32
                if c.ndim == 2 and c.shape[1] == 3:
                    if c.max() <= 1.1: c = (c * 255)
                    c = c.astype(np.uint32)
                    c = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
                
                params["colors"] = c.astype(np.uint32)
            else:
                # DEFAULT COLORMAP MODE
                params["color"] = default_colors[i % len(default_colors)]

            plot += k3d.points(p, **params)
    else:
        global_extent = 1.0

    # -------------------------
    # CAMERAS (HEAVILY OPTIMIZED)
    # -------------------------
    if camera_input is not None:

        if not isinstance(camera_input, (list, tuple)):
            camera_input = [camera_input]

        cam_size = global_extent * 0.12

        all_verts = []
        all_inds = []
        optical_centers = []

        offset = 0

        for P in camera_input:

            C, R, U, F = get_orthogonal_camera_vectors(P)

            C = C.astype(np.float32)
            R = R.astype(np.float32)
            U = U.astype(np.float32)
            F = F.astype(np.float32)

            dist = cam_size
            w, h = cam_size * 0.8, cam_size * 0.6

            c1 = C + F*dist + R*w + U*h
            c2 = C + F*dist - R*w + U*h
            c3 = C + F*dist - R*w - U*h
            c4 = C + F*dist + R*w - U*h

            verts = np.vstack([C, c1, c2, c3, c4]).astype(np.float32)

            # Frustum lines
            inds = np.array([
                [0,1],[0,2],[0,3],[0,4],
                [1,2],[2,3],[3,4],[4,1]
            ], dtype=np.uint32)

            # Tripod axes
            a_len = cam_size * 0.4

            tripod_verts = np.vstack([
                C, C + R*a_len,
                C, C + U*a_len,
                C, C + F*a_len
            ]).astype(np.float32)

            tripod_inds = np.array([
                [5,6],
                [7,8],
                [9,10]
            ], dtype=np.uint32)

            # Merge verts
            merged = np.vstack([verts, tripod_verts])

            all_verts.append(merged)
            all_inds.append(np.vstack([inds, tripod_inds]) + offset)

            offset += merged.shape[0]

            optical_centers.append(C)

        # SINGLE draw call for ALL cameras
        plot += k3d.lines(
            np.vstack(all_verts),
            np.vstack(all_inds),
            color=0xff0000,
            width=float(cam_size * 0.02),
            shader="simple"
        )

        # ONE object for all optical centers
        plot += k3d.points(
            np.vstack(optical_centers),
            point_size=float(cam_size * 0.12),
            color=0x000000,
            shader="flat"
        )

    plot.display()


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

    # Auto grid size if not provided
    if nrows is None or ncols is None:
        ncols = int(math.ceil(math.sqrt(frames_to_plot)))
        nrows = int(math.ceil(frames_to_plot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) or axes.ndim > 0 else [axes]

    total_slots = len(axes)

    for i in range(frames_to_plot):
        idx = start_frame + i
        if idx >= total_frames:
            break

        ax = axes[i]
        ax.imshow(depth[idx, ...])
        if obs_mat is not None and features is not None:
            ax.scatter(
                obs_mat[idx * 2, features],
                obs_mat[idx * 2 + 1, features],
                s=6,
                c="red",
            )
        ax.set_title(f"Frame {idx + 1}, max depth: {depth[idx].max():.2f}")
        ax.axis("off")

    # Hide unused subplots
    for j in range(frames_to_plot, total_slots):
        axes[j].axis("off")

    plt.tight_layout()
    fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    if show:
        plt.show()

    return fig, axes
