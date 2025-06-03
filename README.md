# trackers

Prepare the environment:

	conda create -n trackers -y python=3.11
	conda activate trackers
	pip install -r requirements.txt
Run the tracker and generate a .mat file:

    python3 cotracker.py --online_mode --video_path ../mast3r/images_in/apple.mp4
