import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.block_reconstruction import make_blocks


def plot_matrix_blocks(
    lambda_mat: torch.Tensor,
    block_size: int = 4,
    overlap: int = 2,
    ax=None,
    title: str = "Observation matrix with blocks",
    point_subsample: int = 1,
):
    """
    Plot the observation matrix (observed vs missing) with coloured rectangles
    showing block boundaries and overlap regions.

    Args:
        lambda_mat     : (F, P) depth matrix — NaN where missing
        block_size     : block size passed to make_blocks
        overlap        : overlap passed to make_blocks
        ax             : existing matplotlib Axes to draw on (creates new figure if None)
        title          : plot title
        point_subsample: subsample every N points for display (speeds up large matrices)
    """
    lam = lambda_mat.cpu()
    if point_subsample > 1:
        lam = lam[:, ::point_subsample]

    F, P = lam.shape
    observed = torch.isfinite(lam).numpy()   # (F, P) bool

    blocks = make_blocks(F, block_size=block_size, overlap=overlap)

    if ax is None:
        fig_w = max(10, P / 30)
        fig_h = max(4,  F / 8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure

    # --- Observation matrix as image ---
    # 1 = observed (light blue), 0 = missing (light grey)
    img = observed.astype(np.float32)
    ax.imshow(
        img,
        aspect='auto',
        cmap='Blues',
        vmin=-0.5, vmax=1.2,
        interpolation='none',
        origin='upper',
    )

    # --- Block rectangles ---
    n_blocks = len(blocks)
    cmap_blocks = plt.cm.tab10

    for i, frame_ids in enumerate(blocks):
        f_start = frame_ids[0]
        f_end   = frame_ids[-1]

        color = cmap_blocks(i % 10)

        # Full block boundary
        rect = patches.Rectangle(
            (-0.5, f_start - 0.5),
            P,
            f_end - f_start + 1,
            linewidth=1.5,
            edgecolor=color,
            facecolor='none',
            linestyle='-',
            zorder=3,
        )
        ax.add_patch(rect)

        # Overlap with next block (dashed fill)
        if i < n_blocks - 1:
            next_frame_ids = blocks[i + 1]
            overlap_start = next_frame_ids[0]
            overlap_end   = f_end
            if overlap_end >= overlap_start:
                overlap_rect = patches.Rectangle(
                    (-0.5, overlap_start - 0.5),
                    P,
                    overlap_end - overlap_start + 1,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=color,
                    alpha=0.12,
                    zorder=2,
                )
                ax.add_patch(overlap_rect)

        # Block label on the left
        mid_frame = (f_start + f_end) / 2
        ax.text(
            -1.5, mid_frame,
            f"B{i}",
            ha='right', va='center',
            fontsize=7,
            color=color,
            fontweight='bold',
        )

    # --- Axes ---
    ax.set_xlim(-0.5, P - 0.5)
    ax.set_ylim(F - 0.5, -0.5)   # origin upper-left
    ax.set_xlabel("Point index", fontsize=9)
    ax.set_ylabel("Frame index", fontsize=9)
    ax.set_title(title, fontsize=10)

    # Tick density
    x_step = max(1, P // 10)
    y_step = max(1, F // 10)
    ax.set_xticks(range(0, P, x_step))
    ax.set_yticks(range(0, F, y_step))
    ax.tick_params(labelsize=7)

    # --- Legend ---
    legend_elements = [
        patches.Patch(facecolor=plt.cm.Blues(0.8), label='observed'),
        patches.Patch(facecolor=plt.cm.Blues(0.1), label='missing'),
        patches.Patch(facecolor='none', edgecolor='gray', linestyle='-',
                      linewidth=1.5, label='block boundary'),
        patches.Patch(facecolor='gray', alpha=0.2, label='overlap region'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
              framealpha=0.8, borderpad=0.5)

    # --- Stats annotation ---
    obs_pct = observed.mean() * 100
    ax.text(
        0.01, 0.01,
        f"F={F}  P={P}  observed={obs_pct:.1f}%  "
        f"blocks={n_blocks}  size={block_size}  overlap={overlap}",
        transform=ax.transAxes,
        fontsize=7,
        color='dimgray',
        va='bottom',
    )

    fig.tight_layout()
    return fig, ax
