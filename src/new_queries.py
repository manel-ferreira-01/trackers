import time
import torch

def add_new_tracks(tracks_step, query_pts, px_threshold=5, border_margin=20):
  """
    Add new tracks to parts of the query points that are not covered
    by the current tracks and are not too close to the image borders.
    This version treats the border margin as a percentage of the image dimensions,
    where the image dimensions are computed from the maximums of the query points grid.
    
    Args:
      tracks_step: tensor of shape (1, time, N, 2) with the current tracks.
      query_pts: tensor of shape (1, N, 3) with the query points. 
                 Assumes columns 1 and 2 hold x and y coordinate values.
      px_threshold: distance threshold in pixels to consider a point as covered.
      border_margin: margin as a percentage (0-100) of the computed image dimensions.
      
    Returns:
      new_tracks: tensor of shape (1, time, M, 2) with the new tracks, or None if no valid points.
  """
  # Mark query points that are already covered by current tracks.
  covered_points = torch.zeros(query_pts.shape[1], dtype=torch.bool, device=query_pts.device)
  for i in range(query_pts.shape[1]):
    dist = torch.norm(query_pts[0, i, 1:3] - tracks_step[0, 0, :, :], dim=1)
    if (dist < px_threshold).sum() > 0:
      covered_points[i] = True

  # Select query points that are not already covered.
  new_query_pts = query_pts[0, ~covered_points, :].unsqueeze(0)  # shape (1, M, 3)
  if new_query_pts.shape[1] == 0:
    return None

  # Compute image dimensions as the max values from the grid.
  width = query_pts[0, :, 1].max().item()
  height = query_pts[0, :, 2].max().item()

  x = new_query_pts[0, :, 1]
  y = new_query_pts[0, :, 2]

  margin_x = width * border_margin / 100.0
  margin_y = height * border_margin / 100.0

  border_filter = (x > margin_x) & (x < (width - margin_x)) & \
                  (y > margin_y) & (y < (height - margin_y))
  
  new_query_pts = new_query_pts[:, border_filter, :]

  if new_query_pts.shape[1] == 0:
    return None
  else:
    return new_query_pts
