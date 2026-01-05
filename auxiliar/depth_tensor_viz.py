import math
import matplotlib.pyplot as plt
import numpy as np
import k3d
import torch

def k3d_3d_plot(point_input, scale=70):
    plot = k3d.plot()
    
    colors = [
        0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 
        0x00ffff, 0xff00ff, 0xffa500, 0x800080
    ]

    # --- Handling Input Types ---
    # 1. If it's not a list or tuple, wrap it so we can iterate over it consistently.
    if not isinstance(point_input, (list, tuple)):
        point_list = [point_input]
    else:
        point_list = point_input

    for i, points in enumerate(point_list):
        # 2. Handle PyTorch Tensors -> Convert to NumPy
        if torch is not None and isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        # Ensure it is treated as a numpy array (handles plain lists too)
        points = np.asanyarray(points)

        # 3. Standard Logic
        color = colors[i % len(colors)]
        
        mins = points.min()
        maxs = points.max()
        extent = (maxs - mins)
        
        # Avoid division by zero if extent is 0
        point_size = float(extent / scale) if extent != 0 else 1.0

        # Assuming input is (3, N) -> Transpose to (N, 3) for K3D
        points_obj = k3d.points(
            positions=points.T, 
            point_size=point_size, 
            color=color,
            name=f"Plot {i+1}"
        )
        
        plot += points_obj

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
