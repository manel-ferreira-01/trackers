import sys
sys.path.append('../LightGlue')

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, match_pair 
from lightglue.utils import load_image, rbd

import torch
from tqdm import tqdm
from lightglue.utils import rbd

import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

_MATCHER_CACHE = {}

def _get_matcher(device='cuda'):
    if 'models' not in _MATCHER_CACHE:
        _MATCHER_CACHE['models'] = {
            'extractor': SuperPoint(max_num_keypoints=None).eval().to(device),
            'matcher': LightGlue(features='superpoint').eval().to(device),
        }
    return _MATCHER_CACHE['models']['extractor'], _MATCHER_CACHE['models']['matcher']


def build_anchor_observation_matrix(video_tensor, device='cuda'):
    extractor, matcher = _get_matcher(device)

    num_frames = video_tensor.shape[0]
    
    all_feats = []
    for i in range(num_frames):
        img = (video_tensor[i].permute(2, 0, 1).to(device) + 1) / 2
        all_feats.append(extractor.extract(img))

    feats0 = all_feats[0]
    num_kpts0 = feats0['keypoints'].shape[1]
    
    tracks = torch.full((num_frames, num_kpts0), -1, dtype=torch.long, device=device)
    tracks[0] = torch.arange(num_kpts0, device=device)

    for f in range(1, num_frames):
        feats_f = all_feats[f]
        res = matcher({'image0': feats0, 'image1': feats_f})
        res = rbd(res)
        matches = res['matches']
        tracks[f, matches[:, 0]] = matches[:, 1]

    mask = (tracks != -1).all(dim=0)
    valid_tracks = tracks[:, mask]
    num_pts = valid_tracks.shape[1]

    if num_pts < 4:
        raise ValueError(f"Only {num_pts} points survived. Try reducing SuperPoint threshold.")

    W = torch.zeros((2 * num_frames, num_pts), device=device)
    for f in range(num_frames):
        kpts_f = all_feats[f]['keypoints'][0]
        selected_points = kpts_f[valid_tracks[f]]
        W[2*f]     = selected_points[:, 0]
        W[2*f + 1] = selected_points[:, 1]

    return W

def build_combinatory_observation_matrix(video_tensor, extractor, matcher, device='cuda'):
    """
    Builds a sparse observation matrix by sequentially matching frames.
    Features that drop out or appear later are filled with NaNs.
    
    Returns:
        W: (2 * num_frames, num_unique_tracks) Interleaved [x0, y0, x1, y1...] with NaNs.
    """
    num_frames = video_tensor.shape[0]
    
    # 1. Pre-extract all features
    all_feats = []
    for i in tqdm(range(num_frames), desc="Extracting features"):
        # video_tensor is [-1, 1], LightGlue wants [0, 1]
        img = (video_tensor[i].permute(2, 0, 1).to(device) + 1) / 2
        all_feats.append(extractor.extract(img))

    # 2. Sequential Matching to Build Track Chains
    tracks = [] # Will hold 1D tensors of length num_frames
    
    # Maps a keypoint index in the 'previous' frame to its global track_id
    prev_kpt_to_track_id = {}

    for f in tqdm(range(num_frames - 1), desc="Sequential Matching (Chaining)"):
        feats_prev = all_feats[f]
        feats_curr = all_feats[f+1]
        
        # Match Frame f -> Frame f+1
        res = matcher({'image0': feats_prev, 'image1': feats_curr})
        res = rbd(res)
        matches = res['matches'] # (K, 2)
        
        curr_kpt_to_track_id = {}
        
        for m in matches:
            idx_prev = m[0].item()
            idx_curr = m[1].item()
            
            if idx_prev in prev_kpt_to_track_id:
                # The point exists in an active track; extend it to the current frame
                track_id = prev_kpt_to_track_id[idx_prev]
                tracks[track_id][f+1] = idx_curr
                curr_kpt_to_track_id[idx_curr] = track_id
            else:
                # The point wasn't tracked previously; start a brand new track
                new_track = torch.full((num_frames,), -1, dtype=torch.long, device=device)
                new_track[f] = idx_prev
                new_track[f+1] = idx_curr
                
                track_id = len(tracks)
                tracks.append(new_track)
                curr_kpt_to_track_id[idx_curr] = track_id
                
        # Move forward in time
        prev_kpt_to_track_id = curr_kpt_to_track_id

    if len(tracks) == 0:
        raise ValueError("No matches found across any frames.")

    # Convert tracks list to a 2D tensor of shape (num_frames, num_unique_tracks)
    tracks_tensor = torch.stack(tracks, dim=0).T 
    num_unique_tracks = tracks_tensor.shape[1]

    print(f"Successfully chained {num_unique_tracks} unique feature tracks.")

    # 3. Construct the Observation Matrix W with NaNs
    # Initialize entirely with NaNs
    W = torch.full((2 * num_frames, num_unique_tracks), float('nan'), device=device)
    
    for f in range(num_frames):
        kpts_f = all_feats[f]['keypoints'][0] # (N, 2)
        
        # Find which tracks actually have a recorded keypoint in frame f
        valid_mask = tracks_tensor[f] != -1
        valid_track_indices = torch.where(valid_mask)[0]
        
        if len(valid_track_indices) > 0:
            kpt_indices = tracks_tensor[f, valid_track_indices]
            selected_points = kpts_f[kpt_indices] # (V, 2)
            
            # Fill X coordinates
            W[2*f, valid_track_indices] = selected_points[:, 0]
            # Fill Y coordinates
            W[2*f + 1, valid_track_indices] = selected_points[:, 1]

    return W

from src.tapnext_infer import run_tapnext, init_alltracker
from src.new_queries import add_new_tracks
from auxiliar.read_video import resize_to_max_side
import utils.saveload
import utils.basic
import utils.improc
import torch.nn.functional as F
import sys



def track_video(video_tensor, query_points_initial,
                mode="trackon2",
                device='cpu'):
    
    match mode:

        case "trackon2":
            sys.path.append("../track_on")
            from model.trackon_predictor import Predictor
            from utils.train_utils import load_args_from_yaml

            model_args = load_args_from_yaml("/home/manuelf/track_on/config/test_dinov2.yaml")
            model = Predictor(model_args, checkpoint_path="/home/manuelf/track_on/trackon2_dinov2_checkpoint.pt").to(device).eval()

            traj, vis = model(video_tensor.permute(0,1,4,2,3).to(device), query_points_initial.to(device))

            mask = ~vis.bool()
            traj[mask.unsqueeze(-1).expand_as(traj)] = float('nan')
            output_tensor = traj
            
        # ==============================================================
        case "alltracker":

            model = init_alltracker(device)

            rgbs = resize_to_max_side((video_tensor.permute(0,1,4,2,3) + 1) / 2 * 255)
            B, T, C, H, W = rgbs.shape
            
            flows_e, visconf_maps_e, _, _ = model.forward_sliding(
                rgbs[:, 0:], iters=4, sw=None, is_training=False
            )

            # grid in pixel coords
            grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device="cpu").float()  # [1,H*W,2]
            grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)  # [1,1,2,H,W]

            # flows_e: [1, T, 2, H, W]
            # grid_xy: [1, 1, 2, H, W]
            traj_maps_e = flows_e + grid_xy  # [1, T, 2, H, W]

            # → [1, T, H, W, 2]
            traj_maps_e = traj_maps_e.permute(0, 1, 3, 4, 2)

            # get forward visibility confidence (channel 1)
            vis_forward = visconf_maps_e[:, :, 1, :, :]  # [1, T, H, W]

            # apply a threshold (e.g. > 0.5)
            visible_mask = vis_forward > 0.75             # [1, T, H, W]
            visible_mask = visible_mask.unsqueeze(-1)    # [1, T, H, W, 1]
            visible_mask = visible_mask.expand_as(traj_maps_e)  # [1, T, H, W, 2]

            traj_maps_e[~visible_mask] = float('nan')


            # get original video shape
            H_orig, W_orig = video_tensor.shape[2:4]
            scale_x = W_orig / W  # width ratio
            scale_y = H_orig / H  # height ratio
            scale = torch.tensor([scale_x, scale_y], device=traj_maps_e.device)

            # apply before flattening
            traj_maps_e = traj_maps_e * scale

            # flatten to [1, T, H*W, 2]
            output_tensor = traj_maps_e.reshape(1, traj_maps_e.shape[1], H * W, 2)  # [1, T, N, 2]

        # ==============================================================
        case "tapnext":
            
            tapnext = init_tapnext(device, "/home/manuelf/storage/bootstapnext_ckpt.npz")
            print("Model initialized")

            video_permuted = video_tensor.clone().permute(0, 4, 1, 2, 3) 
            target_frames = video_permuted.shape[2] # Keep frames as 17
            target_height = 256
            target_width = 256
            resized_video = F.interpolate(
                video_permuted, 
                size=(target_frames, target_height, target_width), 
                mode='trilinear',  # Standard for 5D tensors
                align_corners=False
            )
            final_video = resized_video.permute(0, 2, 3, 4, 1)

            query_points_tapnext = query_points_initial.clone()
            query_points_tapnext[0, :, 1:3] /= torch.tensor([video_tensor.shape[3] / final_video.shape[3], video_tensor.shape[2] / final_video.shape[2]]).to(device) # 1, N, 3(frames,x,y)

            track_histories = run_tapnext(
                final_video.to(device),  # send the video resized to the model size
                query_points_tapnext,
                tapnext,
                device=device,
                new_tracks_flag=False,  # or None
            )
            output = {}

            for track_id, trajectory in tqdm(track_histories.items()):  # per feature
                coords = []
                for t in range(num_frames):  # per frame
                    step = next((pos for (frame, pos) in trajectory if frame == t), None)
                    coords.append(step if step is not None else torch.tensor([float('nan'), float('nan')]))
                output[track_id] = torch.stack(coords)  # shape: [num_frames, 2]

            output_list = [trajectory.unsqueeze(0) for _, trajectory in output.items()]
            output_tensor = torch.cat(output_list, dim=0).unsqueeze(0).permute(0, 2, 1, 3)  # [1, num_frames, num_feats, 2]

            # Flip x/y and scale back to original video size
            output_tensor = output_tensor[:, :, :, [1, 0]]
            output_tensor *= torch.tensor([
                video_tensor.shape[3] / final_video.shape[3],
                video_tensor.shape[2] / final_video.shape[2]
            ])

            torch.cuda.empty_cache()

        # ==============================================================
        case "cotracker":
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

            pred_tracks, pred_visibility = cotracker(
                video_tensor.squeeze().permute(0, 3, 1, 2).unsqueeze(0).float().to(device),
                grid_size=step,
                thr=0.9,
            )  # B T N 2,  B T N 1

            pred_tracks = pred_tracks.cpu()
            pred_visibility = pred_visibility.cpu()

            # Ensure shape [1, T, N, 1]
            if pred_visibility.ndim == 3:
                pred_visibility = pred_visibility.unsqueeze(-1)

            # Visibility mask
            visible_mask = pred_visibility.expand_as(pred_tracks)
            pred_tracks_masked = pred_tracks.clone()
            pred_tracks_masked[~visible_mask] = float('nan')

            # Optional flip if you want consistency
            # pred_tracks_masked = pred_tracks_masked[:, :, :, [1, 0]]

            output_tensor = pred_tracks_masked  # [1, num_frames, num_feats, 2]
            torch.cuda.empty_cache()

        # ==============================================================
        case _:
            raise ValueError(f"Unknown tracking mode: {mode}")
        
    torch.cuda.empty_cache()

    # output_tensor_filtered: [1, num_frames, num_feats, 2]
    num_frames = output_tensor.shape[1]
    num_feats = output_tensor.shape[2]

    # Build observation matrix: [feats*2, frames]
    obs_mat_full = torch.full((num_frames * 2, num_feats), float("nan"), device=output_tensor.device)

    for frame in range(num_frames):  # per frame
        obs_mat_full[frame*2, :] = output_tensor[0, frame, :, 0]  # x
        obs_mat_full[frame*2+1, :] = output_tensor[0, frame, :, 1]  # y

    # remove columns (features) that have any NaN values
    valid_columns_mask_no_nan = ~torch.isnan(obs_mat_full).any(dim=0)
    obs_mat = obs_mat_full[:, valid_columns_mask_no_nan]

    return obs_mat


def view_matches(video_tensor, obs_mat):

    # Assuming obs_mat: [2*num_frames, num_unique_tracks]
    # Assuming video_tensor: [batch, num_frames, H, W, C] in [-1, 1]

    num_frames = video_tensor.shape[1]
    num_total_tracks = obs_mat.shape[1]

    # --- Calculate dynamic grid size ---
    cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)

    # Scale figsize based on the number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # Flatten axes 
    if num_frames == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # --- Random Feature Selection ---
    num_features_to_plot = 15  # Change this to see more or fewer points
    # Randomly select unique track IDs and sort them
    features = torch.randperm(num_total_tracks)[:num_features_to_plot]
    features, _ = torch.sort(features)
    features = torch.cat((features, torch.tensor([410]) ))  # Duplicate for x and y

    print(f"Randomly selected {num_features_to_plot} feature tracks:", features.tolist())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))

    # Loop through EVERY frame in the video
    for frame_idx in range(num_frames):
        # Convert frame for plotting and clamp to [0, 1]
        frame_img = ((video_tensor[0, frame_idx].cpu() + 1) / 2.0).clamp(0, 1).numpy()
        
        axes[frame_idx].imshow(frame_img)
        
        # Loop through each selected feature and plot it
        for f_idx, color in zip(features, colors):
            x = obs_mat[frame_idx * 2, f_idx].item()
            y = obs_mat[frame_idx * 2 + 1, f_idx].item()
            
            # Only plot if the coordinate is not NaN
            if not np.isnan(x) and not np.isnan(y):
                axes[frame_idx].scatter(x, y, s=50, color=color, edgecolors='white', linewidth=1.5, zorder=3)
                axes[frame_idx].text(x + 8, y + 8, str(f_idx.item()), color=color, 
                                    fontsize=12, weight='bold', 
                                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        axes[frame_idx].set_title(f"Frame {frame_idx}")
        axes[frame_idx].axis("off")

    # Hide any extra empty subplots if the grid is larger than num_frames
    for extra_idx in range(num_frames, len(axes)):
        axes[extra_idx].axis("off")

    plt.tight_layout()
    plt.show()
