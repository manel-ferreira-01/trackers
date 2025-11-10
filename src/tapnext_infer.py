# tracker.py

import torch
from tqdm import tqdm
from src.new_queries import add_new_tracks
import sys
sys.path.append("/home/manuelf/tapnet")
from tapnet.tapnext.tapnext_torch import TAPNext # type: ignore
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint # type: ignore

import sys, torch
sys.path.append("/home/manuelf/alltracker")
from nets.alltracker import Net

def init_tapnext(device):

  tapnext = TAPNext(image_size=(256, 256)).to(device)

  #set model to eval, not backprop
  tapnext.eval()
  for p in tapnext.parameters():
    p.requires_grad = False

  tapnext = restore_model_from_jax_checkpoint(tapnext, "/home/manuelf/tapnet/tapnet/tapnext/tapnet/checkpoints/bootstapnext_ckpt.npz")

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
