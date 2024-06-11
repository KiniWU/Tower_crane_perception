import torch
import cv2
import numpy as np
from pathlib import Path
from tracking import KF_tracking, KF_init 

# Model
model_path = '/home/haochen/HKCRC/TowerCrane_Test/2D/runs/train/good/weights/best.pt'
model = torch.hub.load('/home/haochen/HKCRC/TowerCrane_Test/2D', 'custom', path=model_path, source='local')

# Images
# video_path = Path("/home/haochen/HKCRC/tower_crane_data/site_data/test3/camera1_for_yolo_200/valid/images")
video_path = Path("/home/haochen/HKCRC/tower_crane_data/site_data/test3/sync_camera_lidar/camera1")
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
# image_list = sorted(video_path.rglob("*.jpg"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]) \
                    if is_int(str(a).split("_")[-1].split(".")[0]) else float('inf'))
# print(image_list)
save_path = Path("yolo_results")
save_path.mkdir(exist_ok=True, parents=True)

# size = (5472, 3648)
size = (3840, 2160)
out = cv2.VideoWriter(str(save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

x_ratio = 3840 / 1344
y_ratio = 2160 / 768

# Inference
for n, i_p in enumerate(image_list):
    # print(i_p)
    print(n)
    img = cv2.imread(str(i_p))
    img_input = cv2.resize(img, (1344, 768), cv2.INTER_CUBIC)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_input, size=1344)

    pred = results.pred[0].cpu().numpy()
    # print("yolo pred\n",pred)

    # pixel_pt = ((pred[0, 0]+pred[0, 2])*x_ratio/2, (pred[0, 1]+pred[0, 3])*y_ratio/2)
    x_pixel  = np.float32((pred[0, 0]+pred[0, 2])*x_ratio/2)
    y_pixel  = np.float32((pred[0, 1]+pred[0, 3])*y_ratio/2)
    pixel_pt = np.array([ [x_pixel], [y_pixel]], dtype=np.float32)
    print("2D_box_center\n",pixel_pt)

    if n == 0:
        initial_state = np.array([pixel_pt[0,0], pixel_pt[1,0], 0, 0], np.float32)
        print("init state\n",initial_state)
        kalman_filter = KF_init(initial_state)

    kalman_filter,corrected_state = KF_tracking(kalman_filter,pixel_pt)
        
    print("2D_box_center based KF\n",corrected_state)

    img = cv2.rectangle(img, 
                    pt1=(int(pred[0, 0]*x_ratio), int(pred[0, 1]*y_ratio)), 
                    pt2=(int(pred[0, 2]*x_ratio), int(pred[0, 3]*y_ratio)), 
                    color=(0, 0, 255), 
                    thickness=10)
    
    out.write(img)
    if n>100:
        break


