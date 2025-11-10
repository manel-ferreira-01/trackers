import torch
import torch.nn.functional as F

def get_rgb_at_coords_torch(image_tensor, xy_coords):
    """
    image_tensor: (3, H, W) float32 torch tensor, range [0,1]
    xy_coords: (N, 2) tensor of (x, y) coordinates in pixel space
    Returns:
        rgb: (N, 3) interpolated RGB values
    """
    C, H, W = image_tensor.shape
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

    # Normalize coords to [-1, 1]
    x = 2 * xy_coords[:, 0] / (W - 1) - 1
    y = 2 * xy_coords[:, 1] / (H - 1) - 1
    grid = torch.stack([x, y], dim=1).view(1, -1, 1, 2)  # (1, N, 1, 2)

    sampled = F.grid_sample(image_tensor, grid, mode='bilinear', align_corners=True)
    rgb = sampled[0, :, :, 0].T  # shape (N, 3)
    return rgb