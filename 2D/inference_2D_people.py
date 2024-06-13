import torch
import cv2
import numpy as np
from pathlib import Path
from tracking import KF_tracking, KF_init 

# Model
model_path = '/home/haochen/HKCRC/TowerCrane_Test/2D/runs/train/human_v1/weights/best.pt'
model = torch.hub.load('/home/haochen/HKCRC/TowerCrane_Test/2D', 'custom', path=model_path, source='local')

# Images
# video_path = Path("/home/haochen/HKCRC/tower_crane_data/site_data/test4/preprocess_data/sync_camera_lidar/camera")

# def is_int(s):
#     try:
#         int(s)
#         return True
#     except ValueError:
#         return False
    
# # image_list = sorted(video_path.rglob("*.jpg"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
# image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]) \
#                     if is_int(str(a).split("_")[-1].split(".")[0]) else float('inf'))

video_path = Path("/home/haochen/HKCRC/tower_crane_data/site_data/test4/training_data/hkcrc-people-MVS/valid/images")
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
image_list = sorted(video_path.rglob("*.jpg"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]) \
                    if is_int(str(a).split("_")[-1].split(".")[0]) else float('inf'))

# print(image_list)
save_path = Path("yolo_results")
save_path.mkdir(exist_ok=True, parents=True)

# size = (5472, 3648)
# size = (3840, 2160)
size = (684, 456)
out = cv2.VideoWriter(str(save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# Inference
for n, i_p in enumerate(image_list):
    print(i_p)
    # print(n)
    img       = cv2.imread(str(i_p))
    # img_input = cv2.resize(img, (1344, 768), cv2.INTER_CUBIC)
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_input, size=684)

    pred = results.pred[0].cpu().numpy()
    print("yolo pred\n",pred)

    img = cv2.rectangle(img, 
                        pt1=(int(pred[0, 0]), int(pred[0, 1])), 
                        pt2=(int(pred[0, 2]), int(pred[0, 3])), 
                        color=(0, 0, 255), 
                        thickness=5)
    cv2.imwrite(str(save_path) + "/camera_"+ str(n)+ ".jpg", img)

    if n>100:
        break


