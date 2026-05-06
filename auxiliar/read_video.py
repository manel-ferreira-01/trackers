import torch
import numpy as np
import os
from glob import glob
import cv2 as cv
import simplejpeg
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
import decord
import torch.nn.functional as Fnn
import time
# Assuming the previous function is saved in this path as per your import
from auxiliar.resize_crop import preprocess_images_batch

decord.bridge.set_bridge('torch')

import torch.nn.functional as F

def natural_sort_key(s):
    import re
    # Splits the string into a list of strings and integers
    # e.g., "crop_yaw10" -> ["crop_yaw", 10]
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def read_video_or_images(path, mode="crop", device='cpu', target_size=518):
    t0 = time.time()

    if os.path.isdir(path):
        image_files = sorted(glob(os.path.join(path, "*")), key=lambda x: natural_sort_key(os.path.basename(x)))
        def _load_rgb(img_path):
            ext = os.path.splitext(img_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                with open(img_path, 'rb') as fh:
                    return simplejpeg.decode_jpeg(fh.read(), colorspace='RGB')
            img = cv.imread(img_path)
            return cv.cvtColor(img, cv.COLOR_BGR2RGB) if img is not None else None

        # Pass 1: get max dims without storing tensors
        max_h, max_w = 0, 0
        valid_paths = []
        for img_path in image_files:
            img = _load_rgb(img_path)
            if img is None:
                continue
            max_h = max(max_h, img.shape[0])
            max_w = max(max_w, img.shape[1])
            valid_paths.append(img_path)

        if not valid_paths:
            raise ValueError(f"No valid images found in folder: {path}")

        print(f"[timing] pass-1 scan ({len(valid_paths)} images): {time.time() - t0:.2f}s"); t1 = time.time()

        # Pass 2a: load images to tensors
        frames = []
        for img_path in tqdm(valid_paths, desc="Loading images", unit="img"):
            frames.append(torch.from_numpy(_load_rgb(img_path)))  # uint8, RGB

        print(f"[timing] pass-2a load ({len(frames)} images): {time.time() - t1:.2f}s"); t1 = time.time()

        # Pass 2b: pad frames to max dims
        padded_frames = []
        for f in tqdm(frames, desc="Padding images", unit="img"):
            pad_h, pad_w = max_h - f.shape[0], max_w - f.shape[1]
            padded_frames.append(F.pad(f.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0))
        del frames

        print(f"[timing] pass-2b pad: {time.time() - t1:.2f}s"); t1 = time.time()

        video_tensor = torch.stack(padded_frames)  # (F, H, W, 3) uint8
        del padded_frames

        print(f"[timing] torch.stack frames: {time.time() - t1:.2f}s")

    else:
        ctx = gpu(0) if device.startswith("cuda") else cpu(0)
        t1 = time.time()
        vr = VideoReader(path, ctx=ctx)
        print(f"[timing] VideoReader init: {time.time() - t1:.2f}s"); t1 = time.time()
        video_tensor = torch.stack([f for f in vr])  # uint8
        print(f"[timing] decode+stack frames ({len(vr)} frames): {time.time() - t1:.2f}s")
        valid_paths = None

    print("Original video tensor shape (F, H, W, C):", video_tensor.shape)

    t1 = time.time()
    # Derive video_original from uint8 tensor, cast once
    video_original = video_tensor.permute(0, 3, 1, 2).float().div(255).sub(0.5).mul(2).to(device)  # (F, C, H, W)
    del video_tensor
    print(f"[timing] cast+normalize to video_original: {time.time() - t1:.2f}s")

    # model_input re-derived from video_original arithmetically (no extra allocation of raw data)
    model_input = video_original.add(1).div(2)  # [-1, 1] -> [0, 1]

    t1 = time.time()
    # Preprocess in chunks to avoid a second full-size allocation
    chunk_size = 8
    preprocessed_chunks = []
    for i in range(0, model_input.shape[0], chunk_size):
        chunk = preprocess_images_batch(model_input[i:i + chunk_size], target_size=target_size)
        preprocessed_chunks.append(chunk.cpu())
    del model_input
    print(f"[timing] preprocess_images_batch ({model_input.shape[0] if False else len(preprocessed_chunks)} chunks): {time.time() - t1:.2f}s")

    t1 = time.time()
    preprocessed_batch = torch.cat(preprocessed_chunks, dim=0)
    del preprocessed_chunks

    video_preprocessed = preprocessed_batch.sub(0.5).mul(2).permute(0, 2, 3, 1).to(device)  # (F, H, W, C)
    del preprocessed_batch
    print(f"[timing] cat+normalize preprocessed batch: {time.time() - t1:.2f}s")

    print("Preprocessed video tensor shape (F, H, W, C):", video_preprocessed.shape)
    print(f"[timing] total read_video_or_images: {time.time() - t0:.2f}s")

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