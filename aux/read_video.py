import cv2 as cv
import torch
import numpy as np


def read_video_torch(video_path, model_video_size=(256, 256), device='cuda'):
    """
    Reads a video file and resizes it to the specified model input size.
    
    Args:
        video_path (str): Path to the video file.
        model_video_size (tuple): Desired size for the video frames (height, width).
        device (str): Device to load the tensor onto ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Video frames as a tensor of shape (num_frames, 3, height, width).
    """
    # Read the video using OpenCV
    cap = cv.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    video_opencv = np.array(frames, dtype=np.float32)
    video_original = torch.tensor(video_opencv)
    video_resized_opencv = np.array([cv.resize(frame, model_video_size) for frame in video_opencv], dtype=np.float32)
    video_resized = torch.tensor(video_resized_opencv).to(device)

    # Normalize to [-1, 1]
    video_resized.div_(255).sub_(0.5).mul_(2)  # in-place normalization
    video_original.div_(255).sub_(0.5).mul_(2)  # in-place normalization

    return video_resized, video_original