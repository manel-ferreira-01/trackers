import torch
import numpy as np
import mediapy
from scipy.io import savemat
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def generate_frame_image_cv(video, obs_matrix):
    """
    Generate an image for each frame of the video with tracking paths drawn.
    Args:
        video (torch.Tensor): Video tensor of shape (1, T, H, W, 3) with pixel values in [-1, 1].
        obs_matrix (np.ndarray): Observation matrix of shape (2F, P) where
                                  rows are [x0, y0, x1, y1, ..., xF, yF] and columns are points.
    """
    T = video.shape[1]
    P = obs_matrix.shape[1]

    # Reshape to (F, P, 2) for easier indexing
    # obs_matrix rows: [x_f0, y_f0, x_f1, y_f1, ...]
    F = obs_matrix.shape[0] // 2
    tracks = obs_matrix.reshape(F, 2, P).transpose(0, 2, 1)  # (F, P, 2)

    to_video = []

    for frame in tqdm(range(T)):
        img = (((video[0, frame].numpy() + 1) / 2) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for track_idx in range(P):
            current_pos = tracks[frame, track_idx]  # (2,)

            if np.isnan(current_pos).any():
                continue

            start = max(0, frame - 5)
            history_window = tracks[start:frame + 1, track_idx]  # (window, 2)

            valid = ~np.isnan(history_window).any(axis=1)
            history_window = history_window[valid]
            if len(history_window) < 2:
                continue

            for i in range(len(history_window) - 1):
                pt1 = tuple(map(int, history_window[i]))
                pt2 = tuple(map(int, history_window[i + 1]))
                color = tuple(int(c) for c in np.array(plt.cm.RdYlBu(i / len(history_window))[:3]) * 255)
                cv2.line(img, pt1, pt2, color=color, thickness=2)

            pt = tuple(map(int, history_window[-1]))
            cv2.circle(img, pt, radius=2, color=(255, 255, 255), thickness=-1)

        output_dir = './test_images'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f'{output_dir}/tracks_frame_{frame:03d}.png', img)
        to_video.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    mediapy.write_video('tracking_output.mp4', to_video, fps=1)