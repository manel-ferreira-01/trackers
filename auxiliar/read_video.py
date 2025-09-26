import cv2 as cv
import torch
import numpy as np
import os
from glob import glob

def read_video_or_images(path, model_video_size=(256, 256), device='cuda'):
    """
    Reads either a video file or a folder of images, resizes them,
    and converts to a tensor suitable for model input.

    Args:
        path (str): Path to a video file or folder of images.
        model_video_size (tuple): Desired size for the frames (height, width).
        device (str): Device to load the tensor onto ('cpu' or 'cuda').

    Returns:
        video_resized (torch.Tensor): Tensor of shape (num_frames, H, W, 3) normalized to [-1, 1].
        video_original (torch.Tensor): Tensor of shape (num_frames, H, W, 3) normalized to [-1, 1].
    """
    frames = []
    
    if os.path.isdir(path):  
        # Read images from folder (sorted for consistent order)
        image_files = sorted(
            glob(os.path.join(path, "*")),
            key=lambda x: os.path.basename(x)
        )
        for img_path in image_files:
            img = cv.imread(img_path)
            if img is None:
                continue
            frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    else:
        # Read video
        cap = cv.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

    if not frames:
        raise ValueError(f"No frames found in path: {path}")

    video_opencv = np.array(frames, dtype=np.float32)
    video_original = torch.tensor(video_opencv)
    video_resized_opencv = np.array([cv.resize(frame, model_video_size) for frame in video_opencv], dtype=np.float32)
    video_resized = torch.tensor(video_resized_opencv).to(device)

    # Normalize to [-1, 1]
    video_resized.div_(255).sub_(0.5).mul_(2)  # in-place normalization
    video_original.div_(255).sub_(0.5).mul_(2)  # in-place normalization

    return video_resized, video_original