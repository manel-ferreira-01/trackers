import torch
import numpy as np
import os
from glob import glob
import cv2 as cv
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
import decord
import torch.nn.functional as Fnn

decord.bridge.set_bridge('torch')

def read_video_or_images(path, model_video_size=(256, 256), device='cpu'):
    """
    Reads either a video file (with Decord) or a folder of images (with OpenCV),
    resizes them, and converts to a tensor suitable for model input.

    Args:
        path (str): Path to a video file or folder of images.
        model_video_size (tuple): Desired size for the frames (height, width).
        device (str): Device to load the tensor onto ('cpu' or 'cuda').

    Returns:
        video_resized (torch.Tensor): (num_frames, H, W, 3) normalized to [-1, 1].
        video_original (torch.Tensor): (num_frames, H, W, 3) normalized to [-1, 1].
    """
    frames = []

    if os.path.isdir(path):
        # --- Read images from folder ---
        image_files = sorted(glob(os.path.join(path, "*")), key=lambda x: os.path.basename(x))
        for img_path in tqdm(image_files, desc="Reading images", unit="img"):
            img = cv.imread(img_path)
            if img is None:
                continue
            frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb))
        if not frames:
            raise ValueError(f"No valid images found in folder: {path}")
        video_tensor = torch.stack(frames).float()  # (F, H, W, 3)

    else:
        # --- Read video using Decord ---
        ctx = gpu(0) if device.startswith("cuda") else cpu(0)
        vr = VideoReader(path, ctx=ctx)
        num_frames = len(vr)

        frames = []
        for i in tqdm(range(num_frames), desc=f"Reading video: {os.path.basename(path)}", unit="frame"):
            frame = vr[i]  # Already returns a torch tensor (H, W, 3)
            frames.append(frame)
        video_tensor = torch.stack(frames).float()  # (F, H, W, 3)

    # --- Store original video before resizing ---
    video_original = video_tensor.clone()

    # --- Resize all frames ---
    resized_frames = torch.nn.functional.interpolate(
        video_tensor.permute(0, 3, 1, 2),  # (F, 3, H, W)
        size=model_video_size,
        mode="bilinear",
        align_corners=False
    ).permute(0, 2, 3, 1)  # (F, H, W, 3)

    # --- Normalize to [-1, 1] ---
    video_resized = resized_frames.div(255).sub(0.5).mul(2)
    video_original = video_original.div(255).sub(0.5).mul(2)

    # --- Move to device ---
    video_resized = video_resized.to(device)
    video_original = video_original.to(device)

    return video_resized, video_original

def resize_to_max_side(x, max_side=1024):
    """
    x: tensor of shape (1, num_frames, 3, H, W)
    returns: resized tensor with largest spatial side = max_side
    """
    _, num_frames, C, H, W = x.shape
    scale = max_side / max(H, W)
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    
    x = x.view(-1, C, H, W)  # (B*F, 3, H, W)
    x_resized = Fnn.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return x_resized.view(1, num_frames, C, new_H, new_W)