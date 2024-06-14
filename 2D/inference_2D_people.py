import torch
import cv2
import numpy as np
from pathlib import Path
from tracking import KF_tracking, KF_init 
from config import model_human_v1_path, MVS_video_path, Small_human_save_path, img_size


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

from small_human_detection_im_crop import crop_im
# Model

model = torch.hub.load('yolov5', 'custom', path=model_human_v1_path, source='local')
model.iou = 0.2
model.conf = 0.2
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

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
image_list = sorted(MVS_video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]) \
                    if is_int(str(a).split("_")[-1].split(".")[0]) else float('inf'))

# print(image_list)
Small_human_save_path.mkdir(exist_ok=True, parents=True)

# size = (5472, 3648)
# size = (3840, 2160)
size = (1280, 1280)
out = cv2.VideoWriter(str(Small_human_save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# Inference
ori_w, ori_h = img_size

num_batches = (8, 8)


cropped_w = ori_w//num_batches[0]
cropped_h = ori_h//num_batches[1]

x_ratio = cropped_w / size[0] 
y_ratio = cropped_h / size[1]

for n, i_p in enumerate(image_list):
    print(i_p)
    # print(n)
    img = cv2.imread(str(i_p))
    img_detected = np.zeros_like(img, np.uint8)
    img_patches = crop_im(img, num_batches=num_batches)
    for i in range(num_batches[0]):
        x_start = i*cropped_w
        x_end = (i+1)*cropped_w
        if x_end > ori_w:
            x_end = ori_w
        for j in range(num_batches[1]):
            y_start = j*cropped_h
            y_end = (j+1)*cropped_h
            if y_end > ori_h:
                y_end = ori_h

            print(len(img_patches))
            img_piece = img_patches[i*8 + j]
            print(img_piece.shape)
            img_input = cv2.resize(img_piece, size, cv2.INTER_CUBIC)
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            
            # Inference
            results = model(img_input, size=684)

            pred = results.pred[0].cpu().numpy()
            print("yolo pred\n",pred)
            if pred.shape[0] > 0:
                img_piece = cv2.rectangle(img_piece, 
                                    pt1=(int(pred[0, 0]*x_ratio), int(pred[0, 1]*y_ratio)), 
                                    pt2=(int(pred[0, 2]*x_ratio), int(pred[0, 3]*y_ratio)), 
                                    color=(0, 0, 255), 
                                    thickness=5)
            cv2.imwrite(str(Small_human_save_path) + "/camera_"+ str(i) + "_" + str(j) + ".jpg", img_piece)
            img_detected[y_start:y_end, x_start:x_end] = img_piece
    cv2.imwrite(str(Small_human_save_path) + "/camera_"+ str(n)+ ".jpg", img_detected)

    if n>100:
        break


