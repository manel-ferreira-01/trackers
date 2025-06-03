# trackers

Prepare the environment:

	conda create -n trackers -y python=3.11
	conda activate trackers
	pip install -r requirements.txt
Run the tracker and generate a .mat file:

    python3 cotracker.py --online_mode --video_path ../mast3r/images_in/apple.mp4


## options 

      -h, --help            show this help message and exit
      --device {cuda,cpu}   Device to run the model on.
      --online_mode         Use online mode for tracking.
      --grid_size GRID_SIZE
                            Grid size for keypoint initialization.
      --video_output        Save video output visualization.
      --video_path VIDEO_PATH
                            Path to the input video.
      --output_mat OUTPUT_MAT
                            Path to save the output .mat file.
