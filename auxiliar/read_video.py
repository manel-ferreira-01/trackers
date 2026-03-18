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
    if os.path.isdir(path):
        image_files = sorted(glob(os.path.join(path, "*")), key=lambda x: os.path.basename(x))

        # Pass 1: get max dims without storing tensors
        max_h, max_w = 0, 0
        valid_paths = []
        for img_path in image_files:
            img = cv.imread(img_path)
            if img is None:
                continue
            max_h = max(max_h, img.shape[0])
            max_w = max(max_w, img.shape[1])
            valid_paths.append(img_path)

        if not valid_paths:
            raise ValueError(f"No valid images found in folder: {path}")

        # Pass 2: load, pad as uint8, stack
        padded_frames = []
        for img_path in tqdm(valid_paths, desc="Reading images", unit="img"):
            img = cv.imread(img_path)
            f = torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # uint8
            pad_h, pad_w = max_h - f.shape[0], max_w - f.shape[1]
            f_padded = F.pad(f.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0)
            padded_frames.append(f_padded)

        video_tensor = torch.stack(padded_frames)  # (F, H, W, 3) uint8
        del padded_frames

    else:
        ctx = gpu(0) if device.startswith("cuda") else cpu(0)
        vr = VideoReader(path, ctx=ctx)
        video_tensor = torch.stack([f for f in vr])  # uint8
        valid_paths = None

    print("Original video tensor shape (F, H, W, C):", video_tensor.shape)

    # Derive video_original from uint8 tensor, cast once
    video_original = video_tensor.permute(0, 3, 1, 2).float().div(255).sub(0.5).mul(2).to(device)  # (F, C, H, W)
    del video_tensor

    # model_input re-derived from video_original arithmetically (no extra allocation of raw data)
    model_input = video_original.add(1).div(2)  # [-1, 1] -> [0, 1]

    # Preprocess in chunks to avoid a second full-size allocation
    chunk_size = 8
    preprocessed_chunks = []
    for i in range(0, model_input.shape[0], chunk_size):
        chunk = preprocess_images_batch(model_input[i:i + chunk_size], target_size=target_size)
        preprocessed_chunks.append(chunk.cpu())
    del model_input

    preprocessed_batch = torch.cat(preprocessed_chunks, dim=0)
    del preprocessed_chunks

    video_preprocessed = preprocessed_batch.sub(0.5).mul(2).permute(0, 2, 3, 1).to(device)  # (F, H, W, C)
    del preprocessed_batch

    print("Preprocessed video tensor shape (F, H, W, C):", video_preprocessed.shape)

    return video_original, valid_paths

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