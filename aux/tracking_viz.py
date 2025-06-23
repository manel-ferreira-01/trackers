import torch
import numpy as np
import mediapy
from scipy.io import savemat
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def generate_frame_image_cv(video, output_tensor):
    """
    Generate an image for each frame of the video with tracking paths drawn.
    Args:
        video (torch.Tensor): Video tensor of shape (1, T, H, W, 3) with pixel values in [-1, 1].
        output_tensor (torch.Tensor): Output tensor of shape (1, T, N, 2)
    """

    to_video = []

    for frame in tqdm(range(video.shape[1])):
        # Convert image to numpy BGR for OpenCV
        img = (((video[0, frame].numpy() + 1) / 2 )*255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        num_feats = output_tensor.shape[2]
        for track_idx in range(num_feats):
            track_history = output_tensor[0, :, track_idx]
            current_pos = track_history[frame]

            if not torch.isnan(current_pos).any():
                start = max(0, frame - 5)
                end = frame + 1
                history_window = track_history[start:end].cpu().numpy()

                # Remove NaNs
                valid = ~np.isnan(history_window).any(axis=1)
                history_window = history_window[valid]
                if len(history_window) < 2:
                    continue

                # Draw path with color gradient
                for i in range(len(history_window) - 1):
                    pt1 = tuple(map(int, history_window[i]))
                    pt2 = tuple(map(int, history_window[i + 1]))
                    color = tuple(int(c) for c in np.array(plt.cm.RdYlBu(i / len(history_window))[:3]) * 255)
                    cv2.line(img, pt1, pt2, color=color, thickness=2)

                # Draw last point
                pt = tuple(map(int, history_window[-1]))
                cv2.circle(img, pt, radius=2, color=(255, 255, 255), thickness=-1)

        output_dir = './test_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f'{output_dir}/tracks_frame_{frame:03d}.png'
        cv2.imwrite(output_path, img)

        to_video.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save video
    mediapy.write_video('tracking_output.mp4', to_video, fps=30)