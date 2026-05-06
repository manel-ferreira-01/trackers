# tracker.py

import torch
from tqdm import tqdm
from src.new_queries import add_new_tracks
import sys
sys.path.append("/home/manuelf/tapnet")
from tapnet.tapnext.tapnext_torch import TAPNext,TAPNextTrackingState # type: ignore
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint # type: ignore


import sys, torch
sys.path.append("/home/manuelf/alltracker")
from nets.alltracker import Net

def init_tapnext(device, model_path):

  tapnext = TAPNext(image_size=(256, 256)).to(device)

  #set model to eval, not backprop
  tapnext.eval()
  for p in tapnext.parameters():
    p.requires_grad = False

  tapnext = restore_model_from_jax_checkpoint(tapnext, model_path)

  return tapnext

def init_alltracker(device):
    model = Net(16)
    global_step = 0
    # pick tiny or full weights
    if 0:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    print("loaded weights from", url)
    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


def run_tapnext(video_tensor_resized, query_points_initial, tapnext, device="cuda", new_tracks_flag=False):
    """
    Run TAP-NEXT tracking over a video tensor.

    Args:
        video_tensor_resized (torch.Tensor): Input video tensor of shape [B, T, C, H, W].
        query_points_initial (torch.Tensor): Initial query points tensor.
        tapnext (Callable): TAP-NEXT model callable.
        device (str): Device to run on ("cuda" or "cpu").
        new_tracks_flag (bool): Whether to add new tracks during tracking.

    Returns:
        track_histories (dict): track_id -> list of (frame_index, (x, y) or None).
    """

    next_track_id = 0  # unique ID for every track
    track_histories = {}  # track_id -> list of (frame_index, (x, y) or None)
    active_tracks = {}  # current_idx -> track_id (used during current step)

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float32, enabled=True):

            # First frame initialization
            tracks, tracks_logits, visible_logits, tracking_state = tapnext(
                video=video_tensor_resized[:, :1], 
                query_points=query_points_initial
            )

            num_feats = tracks.shape[2]
            for i in range(num_feats):
                track_id = next_track_id
                next_track_id += 1
                active_tracks[i] = track_id
                track_histories[track_id] = [
                    (0, tracks[0, 0, i, :2].cpu())
                ]  # frame 0 position

            # Process remaining frames
            for k in tqdm(range(1, video_tensor_resized.shape[1])):

                tracks_step, tracks_logits_step, visible_logits_step, tracking_state = tapnext(
                    video=video_tensor_resized[:, k:k + 1],
                    state=tracking_state
                )

                visible = (visible_logits_step.squeeze() > 0).cpu()

                for i, track_tensor in enumerate(tracks_step[0, 0]):
                    track_id = active_tracks.get(i, None)
                    if track_id is not None:
                        if visible[i]:
                            track_histories[track_id].append((k, track_tensor[:2].cpu()))
                        else:
                            track_histories[track_id].append((k, None))  # Not visible

                # Optional: Add new features
                if new_tracks_flag:
                    new_tracks = add_new_tracks(tracks_step, query_points_initial)

                    if new_tracks is not None and new_tracks.shape[1] > 0:
                        print(f"Adding new tracks: {new_tracks.shape[1]}")

                        new_tracks[0, :, 0] = 0  # set time to 0 for tapnext reinit

                        # Retain visible active tracks from previous step
                        retained_indices = [i for i, v in enumerate(visible) if v]
                        retained_ids = [active_tracks[i] for i in retained_indices]

                        retained_tracks = tracks_step[0, 0, retained_indices].unsqueeze(0)
                        zero_time = torch.zeros((1, len(retained_indices), 1)).to(retained_tracks.device)
                        retained_tracks = torch.cat([zero_time, retained_tracks], dim=2)

                        concat_tracks = torch.cat([retained_tracks, new_tracks.to(device)], dim=1)

                        tracks_step, tracks_logits_step, visible_logits_step, tracking_state = tapnext(
                            video=video_tensor_resized[:, k].unsqueeze(0),
                            query_points=concat_tracks
                        )

                        # update active_tracks dict
                        active_tracks = {}
                        for i, id in enumerate(retained_ids):
                            active_tracks[i] = id  # preserve old IDs

                        new_start = len(retained_ids)
                        visible = (visible_logits_step.squeeze() > 0).cpu()

                        for i in range(new_tracks.shape[1]):
                            track_id = next_track_id
                            next_track_id += 1
                            active_tracks[new_start + i] = track_id
                            if visible[new_start + i]:
                                track_histories[track_id] = [
                                    (k, tracks_step[0, 0, new_start + i, :2].cpu())
                                ]
                            else:
                                track_histories[track_id] = [(k, None)]

    return track_histories

def make_grid_detector_fn(cell_size=8, min_dist=3, device='cuda'):
    """
    Detects empty grid cells and returns one query point per empty cell.
    cell_size: size of each grid cell in pixels (tapnext 256x256 space)
    """
    def detector_fn(frame_tensor, existing_positions, frame_idx, H=256, W=256):
        # Build grid cell centers
        ys = torch.arange(cell_size // 2, H, cell_size, device=device).float()
        xs = torch.arange(cell_size // 2, W, cell_size, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        cell_centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [M, 2] (x, y)

        if existing_positions.shape[0] > 0:
            # Check which cells already have a point nearby
            dists = torch.cdist(cell_centers.float(), existing_positions.float())  # [M, N]
            occupied = dists.min(dim=1).values < min_dist
            cell_centers = cell_centers[~occupied]

        if cell_centers.shape[0] == 0:
            return None

        t_col = torch.full((cell_centers.shape[0], 1), frame_idx, dtype=torch.float32, device=device)
        return torch.cat([t_col, cell_centers], dim=1)  # [N_new, 3]

    return detector_fn

def run_tapnext_growing(video_tensor, query_points_initial, tapnext, device="cuda"):
    B, T, C, H, W = video_tensor.shape
    next_track_id = 0
    track_histories = {}

    # Each "generation" is a dict with:
    #   'state': current TAPNextTrackingState
    #   'ids': list of track_ids in this generation (in order)
    generations = []
    detector_fn = make_grid_detector_fn(cell_size=8, min_dist=3, device=device)
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.float32, enabled=True):

            # ── Frame 0: initialize first generation ──────────────────
            tracks, _, vis_logits, state0 = tapnext(
                video=video_tensor[:, :1],
                query_points=query_points_initial
            )
            N0 = tracks.shape[2]
            ids0 = []
            for i in range(N0):
                tid = next_track_id; next_track_id += 1
                ids0.append(tid)
                vis = vis_logits[0, 0, i].item() > 0
                track_histories[tid] = [(0, tracks[0, 0, i, :2].cpu() if vis else None)]

            generations.append({'state': state0, 'ids': ids0})

            # ── Frames 1..T-1 ─────────────────────────────────────────
            for k in tqdm(range(1, T)):
                frame = video_tensor[:, k:k+1]

                # Step all generations forward
                all_visible_positions = []
                for gen in generations:
                    tracks_k, _, vis_k, gen['state'] = tapnext(
                        video=frame,
                        state=gen['state']
                    )
                    vis = (vis_k[0, 0] > 0).cpu()  # [N_gen]
                    for i, tid in enumerate(gen['ids']):
                        pos = tracks_k[0, 0, i, :2].cpu()
                        track_histories[tid].append((k, pos if vis[i] else None))
                        if vis[i]:
                            all_visible_positions.append(pos)

                # Detect new points in empty grid cells
                if k % 5 == 0:
                    existing_pos = torch.stack(all_visible_positions).to(device) \
                        if all_visible_positions else torch.zeros((0, 2), device=device)
                    new_queries = detector_fn(frame[0, 0], existing_pos, frame_idx=k)

                    if new_queries is not None and new_queries.shape[0] > 0:
                        N_new = new_queries.shape[0]
                        new_queries_b = new_queries.unsqueeze(0).to(device)  # [1, N_new, 3]

                        # Initialize new generation over full prefix
                        new_tracks_full, _, new_vis_full, new_state = tapnext(
                            video=video_tensor[:, :k+1],
                            query_points=new_queries_b
                        )

                        new_ids = []
                        for i in range(N_new):
                            tid = next_track_id; next_track_id += 1
                            new_ids.append(tid)
                            # Fill frames 0..k-1 as None (not yet born)
                            history = [(f, None) for f in range(k)]
                            vis_i = new_vis_full[0, k, i].item() > 0
                            pos_i = new_tracks_full[0, k, i, :2].cpu()
                            history.append((k, pos_i if vis_i else None))
                            track_histories[tid] = history

                        generations.append({'state': new_state, 'ids': new_ids})

    return track_histories