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
        R_ortho[:,0] *= -1

    # 4. Extract Orthogonal Axes
    # Row 0 = Right, Row 1 = Up (inverted for screen space), Row 2 = Forward
    right = R_ortho[0, :]
    up = -R_ortho[1, :] 
    forward = R_ortho[2, :]
    
    return C, right, up, forward

def k3d_3d_plot(point_input, camera_input=None, scale=70):
    plot = k3d.plot(camera_auto_fit=True)
    
    # High-visibility color palette
    colors = [
        0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 
        0x00ffff, 0xff00ff, 0xffa500, 0x800080
    ]
    
    # --- 1. Process and Plot Points ---
    global_extent = 1.0
    if point_input is not None:
        if not isinstance(point_input, (list, tuple)):
            point_input = [point_input]
        
        pts_list = []
        for p in point_input:
            if torch.is_tensor(p): p = p.detach().cpu().numpy()
            p = p.T if p.shape[0] == 3 else p
            pts_list.append(p)
            
        # Determine scene scale for sizing everything else
        combined = np.vstack(pts_list)
        mins, maxs = combined.min(axis=0), combined.max(axis=0)
        global_extent = np.linalg.norm(maxs - mins)
        if global_extent == 0: global_extent = 1.0

        for i, p in enumerate(pts_list):
            color = colors[i % len(colors)]
            psize = float(global_extent / scale) # FIX: TraitError cast
            
            plot += k3d.points(
                p.astype(np.float32), 
                point_size=psize, 
                color=color,
                #shader="flat",
                name=f"Point Set {i}"
            )

    # --- 2. Process and Plot Cameras ---
    if camera_input is not None:
        if not isinstance(camera_input, (list, tuple)):
            camera_input = [camera_input]
            
        cam_size = global_extent * 0.12 # Base size for the camera geometry
        
        for i, P in enumerate(camera_input):
            C, R, U, F = get_orthogonal_camera_vectors(P)
            
            # Construct the Frustum geometry
            dist = cam_size
            w, h = cam_size * 0.8, cam_size * 0.6
            
            # 4 corners of the image plane
            c1 = C + F*dist + R*w + U*h
            c2 = C + F*dist - R*w + U*h
            c3 = C + F*dist - R*w - U*h
            c4 = C + F*dist + R*w - U*h
            
            corners = np.array([c1, c2, c3, c4], dtype=np.float32)
            verts = np.vstack([C, corners]).astype(np.float32)
            
            # Indices for 8 lines: 4 rays from center + 4 lines for the rectangle
            indices = np.array([
                [0,1], [0,2], [0,3], [0,4], 
                [1,2], [2,3], [3,4], [4,1],[1,2]
            ], dtype=np.uint32)
            
            # Main Frustum Box (Thin white lines)
            plot += k3d.lines(
                verts, indices, 
                color=0xff0000, 
                width=float(cam_size * 0.015),
                name=f"Camera {i} Box"
            )

            # RGB Orientation Tripod at Camera Center
            # (Red: Right, Green: Up, Blue: Forward)
            a_len = cam_size * 0.4
            tripod_width = float(cam_size * 0.04)
            
            plot += k3d.lines(np.vstack([C, C + R*a_len]).astype(np.float32), [[0,1]], color=0xff0000, width=tripod_width)
            plot += k3d.lines(np.vstack([C, C + U*a_len]).astype(np.float32), [[0,1]], color=0x00ff00, width=tripod_width)
            plot += k3d.lines(np.vstack([C, C + F*a_len]).astype(np.float32), [[0,1]], color=0x0000ff, width=tripod_width)

            # Optical Center (small white sphere)
            plot += k3d.points(
                C.reshape(1,3).astype(np.float32), 
                point_size=float(cam_size * 0.1), 
                color=0xffffff
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
