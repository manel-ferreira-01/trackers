import torch
import torch.nn.functional as F

def preprocess_images_batch(image_tensor_batch, mode="crop", target_size=518):
    """
    Preprocess a batch of images for model input.

    Assumes tensor is in (S, C, H, W) format with values in [0, 1].

    Args:
        image_tensor_batch (torch.Tensor): Batch of images (S, C, H, W)
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (S, 3, H, W)

    Raises:
        ValueError: If the input is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - When mode="crop": Ensures width=518px while maintaining aspect ratio,
          and height is center-cropped if larger than 518px
        - When mode="pad": Ensures the largest dimension is 518px while maintaining aspect ratio,
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14
    """
    if image_tensor_batch.ndim != 4:
        raise ValueError("Input must be a 4D tensor (S, C, H, W)")

    if image_tensor_batch.size(0) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    shapes = set()
    processed_images = []

    for img in image_tensor_batch:
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError("Each image must be in (3, H, W) format")

        _, height, width = img.shape

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize
        img = F.interpolate(
            img.unsqueeze(0),                # (1, C, H, W)
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)                         # back to (C, H, W)

        # Center crop height if it's larger than target_size (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to square target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        processed_images.append(img)

    # If different shapes, pad to max shape
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in processed_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        processed_images = padded_images

    return torch.stack(processed_images)
