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

import torch.nn.functional as F

def read_video_or_images(path, mode="crop", device='cpu', target_size=518):
    frames = []

    if os.path.isdir(path):
        # --- Read images from folder ---
        image_files = sorted(glob(os.path.join(path, "*")), key=lambda x: os.path.basename(x))
        
        raw_frames = []
        max_h, max_w = 0, 0
        
        for img_path in tqdm(image_files, desc="Reading images", unit="img"):
            img = cv.imread(img_path)
            if img is None: continue
            
            frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            f_tensor = torch.from_numpy(frame_rgb).float() # (H, W, 3)
            
            # Track max dimensions for padding
            h, w, _ = f_tensor.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            raw_frames.append(f_tensor)
        
        if not raw_frames:
            raise ValueError(f"No valid images found in folder: {path}")

        # --- Pad images to common size (max_h, max_w) ---
        padded_frames = []
        for f in raw_frames:
            h, w, _ = f.shape
            pad_h = max_h - h
            pad_w = max_w - w
            
            # F.pad expects (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
            # We pad trailing edges (Right and Bottom) to maintain top-left alignment
            # Note: We permute to (C, H, W) for padding, then back
            f_padded = F.pad(f.permute(2, 0, 1), (0, pad_w, 0, pad_h), value=0)
            padded_frames.append(f_padded.permute(1, 2, 0))
            
        video_tensor = torch.stack(padded_frames)  # (F, H, W, 3)

    else:
        # --- Read video using Decord (Standard sizing assumed for video streams) ---
        ctx = gpu(0) if device.startswith("cuda") else cpu(0)
        vr = VideoReader(path, ctx=ctx)
        video_tensor = torch.stack([f for f in vr]).float()

    # --- Standardize and Return ---
    video_original = video_tensor.clone().div(255).sub(0.5).mul(2).to(device)

    print("Original video tensor shape (F, H, W, C):", video_tensor.shape)
    # Convert to (F, C, H, W) and scale to [0, 1] for preprocessing
    model_input = video_tensor.permute(0, 3, 1, 2).div(255.0)

    # This function now receives a uniform batch
    preprocessed_batch = preprocess_images_batch(model_input, target_size=target_size)

    video_preprocessed = preprocessed_batch.sub(0.5).mul(2).permute(0, 2, 3, 1)

    print("Preprocessed video tensor shape (F, H, W, C):", video_preprocessed.shape)

    return video_original.to(device), image_files

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