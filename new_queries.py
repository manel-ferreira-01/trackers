import time
import torch

def add_new_tracks(tracks_step, query_pts, px_thereshold=40 ):
  """
    Add new tracks to parts of the query points that are not being covered
    by the current tracks.
    
    Args:
      tracks_step: tensor of shape (1, time, N, 2) with the current tracks
      visible_logits_step: tensor of shape (1, time, N, 2) with the visibility logits
      query_pts: tensor of shape (1, N, 3) with the query points in the original image size
  Returns:
      new_tracks: tensor of shape (1, time, M, 2) with the new tracks
  """

  start = time.time()
  # find the points that are not covered by the current tracks
  covered_points = torch.zeros(query_pts.shape[1], dtype=torch.bool, device=query_pts.device)
  trakcs_to_add = []

  for i in range(query_pts.shape[1]):
    # check if the query point is close to the track point
    dist = torch.norm(query_pts[0, i, 1:3] - tracks_step[0, 0, :, :], dim=1)
    if (dist < px_thereshold).sum() > 0: # if there is one track point within 10 pixels, consider it covered
      covered_points[i] = True

  new_query_pts = query_pts[0, ~covered_points, :].unsqueeze(0)  # shape (1, M, 3)

  #print(f"Time to find new query points: {time.time() - start:.2f} seconds")
  if new_query_pts.shape[1] == 0:
    # no new points to track
    return None  
  else:
    return new_query_pts
  
