import torch
import numpy as np
import mediapy
from scipy.io import savemat
import argparse
from tqdm import tqdm

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    video = torch.tensor(mediapy.read_video(args.video_path)).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()

    # Load model
    if args.online_mode:
        cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        cotracker_model(video_chunk=video, is_first_step=True, grid_size=args.grid_size)

        for ind in tqdm(range(0, video.shape[1] - cotracker_model.step, cotracker_model.step)):
            pred_tracks, pred_visibility = cotracker_model(
                video_chunk=video[:, ind : ind + cotracker_model.step * 2]
            )
    else:
        cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        pred_tracks, pred_visibility = cotracker_model(video, grid_size=args.grid_size)

    # Save .mat file
    savemat(args.output_mat, {
        "tracks": pred_tracks.cpu().numpy(),
        "visibility": pred_visibility.cpu().numpy(),
    })

    # Optional: visualize
    if args.video_output:
        from cotracker.utils.visualizer import Visualizer
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoTracker on a video file.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run the model on.")
    parser.add_argument("--online_mode", action="store_true",
                        help="Use online mode for tracking.")
    parser.add_argument("--grid_size", type=int, default=10,
                        help="Grid size for keypoint initialization.")
    parser.add_argument("--video_output", action="store_true",
                        help="Save video output visualization.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video.")
    parser.add_argument("--output_mat", type=str, default="cotracker_tracks.mat",
                        help="Path to save the output .mat file.")

    args = parser.parse_args()
    main(args)
