from pathlib import Path


model_path = Path('/home/Tower_crane_perception/2D/runs/train/exp/weights/last.pt')
video_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/hikrobot/")
video_with_human_path =  Path("/home/Tower_crane_perception/2D/human_detection/demo_vis")
lidar_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/livox/")
save_path = Path("/home/Tower_crane_perception/2D/runs/inference/2d_lidar_livox/")

