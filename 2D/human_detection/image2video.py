import cv2
from pathlib import Path


video_path = Path("/media/simu/8TB_HDD/Tower_crane_perception/2D/human_detection/demo_vis")

image_list = sorted(video_path.rglob("*.jpg"), key=lambda a: int(str(str(a.name)[7:-4])))

save_path = Path("/media/simu/8TB_HDD/Tower_crane_perception/2D/runs/inference/2d_lidar_livox_human/")
save_path.mkdir(exist_ok=True, parents=True)

size = (5472, 3648)

out = cv2.VideoWriter(str(save_path / "video_small_human.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True) 

print(image_list)
for n, i_p in enumerate(image_list):
    img = cv2.imread(str(i_p))
    out.write(img)

# vis.destroy_window()
cv2.destroyAllWindows()
out.release()