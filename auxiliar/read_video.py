import torch
import numpy as np
import os
from glob import glob
import cv2 as cv
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
import decord
import torch.nn.functional as Fnn
# Assuming the previous function is saved in this path as per your import
from auxiliar.resize_crop import preprocess_images_batch

decord.bridge.set_bridge('torch')

def read_video_or_images(path, mode="crop", device='cpu', target_size=518):
    """
    Reads a video or images, and preprocesses them using the specific
    DINOv2-style preprocessing (518px resolution).

    Args:
        path (str): Path to a video file or folder of images.
        mode (str): "pad" (preserves aspect ratio) or "crop" (square center crop).
        device (str): Device to load the tensor onto ('cpu' or 'cuda').

    Returns:
        video_preprocessed (torch.Tensor): (num_frames, H, W, 3) normalized to [-1, 1].
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
            # OpenCV reads BGR, convert to RGB
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

        # Decord returns (F, H, W, 3) directly if using batch retrieval, 
        # but looping is safer for memory if needed. 
        # Here we stick to your loop pattern for consistency.
        frames = []
        for i in tqdm(range(num_frames), desc=f"Reading video: {os.path.basename(path)}", unit="frame"):
            frame = vr[i] 
            frames.append(frame)
        
        video_tensor = torch.stack(frames).float()  # (F, H, W, 3)

    # --- Store original video (Normalize to [-1, 1] for return) ---
    # Clone because the next steps will modify data
    video_original = video_tensor.clone()
    video_original = video_original.div(255).sub(0.5).mul(2).to(device)

    # --- Prepare for Preprocessing ---
    # 1. Convert to (F, C, H, W) -> Required by preprocess_images_batch
    # 2. Normalize to [0, 1] -> Required by preprocess_images_batch
    model_input = video_tensor.permute(0, 3, 1, 2).div(255.0)

    # --- Apply the Preprocessing Logic ---
    # This handles the resizing to 518px, padding/cropping, and ensures divisibility by 14
    # We do this on CPU to avoid OOM on GPU for large batches, unless you have plenty of VRAM
    preprocessed_batch = preprocess_images_batch(model_input, target_size=target_size)

    # --- Finalize Output ---
    # 1. Convert [0, 1] back to [-1, 1] (as per your original function's logic)
    video_preprocessed = preprocessed_batch.sub(0.5).mul(2)

    # 2. Permute back to (F, H, W, 3) (Channel Last)
    video_preprocessed = video_preprocessed.permute(0, 2, 3, 1)

    return video_preprocessed.to(device), video_original

def resize_to_max_side(x, max_side=1024):
    """
    Helper function (kept unchanged).
    x: tensor of shape (1, num_frames, 3, H, W)
    returns: resized tensor with largest spatial side = max_side
    """
    _, num_frames, C, H, W = x.shape
    scale = max_side / max(H, W)
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    
    x = x.view(-1, C, H, W)
    x_resized = Fnn.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return x_resized.view(1, num_frames, C, new_H, new_W)