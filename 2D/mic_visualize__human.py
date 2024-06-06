import torch
import cv2
from pathlib import Path
from tower_utils import pixel2Camera, camera2Lidar, find_closest_cluster, get_3d_box_from_points, lidar2Camera, camera2Pixel
import open3d as o3d
import time
import numpy as np

from tracking import tracking_diff
from config import model_path, video_path, video_with_human_path, save_path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

from pointCloud_clustering import pcd_clustering

print(sys.path)

ORI_RESO = True
# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.iou = 0.2
model.conf = 0.4
# Images
image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
image_human_list = sorted(video_with_human_path.rglob("*.jpg"), key=lambda a: int(str(str(a.name)[7:-4])))
save_path.mkdir(exist_ok=True, parents=True)

if ORI_RESO:
    size = (5472, 3648)
else:
    size = (1344, 896)
out = cv2.VideoWriter(str(save_path / "video_comp_mic_human.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True) 


x_ratio = 5472 / 1344
y_ratio = 3648 / 896
last_pred = None
start_tracking = False
end_tracking = False
last_detected = False
not_detected_cnt = 0
not_detected_THRE = 20

previsous_pts = []
threed_boxes = []

font = cv2.FONT_HERSHEY_SIMPLEX 
# fontScale 
fontScale = 5
   
# Blue color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 10

print(image_list, image_human_list)
for n, (i_p, i_h_p) in enumerate(zip(image_list[:50], image_human_list)):
    print(i_p, i_h_p)
    img = cv2.imread(str(i_p))
    img_human = cv2.imread(str(i_h_p))
    img_input = cv2.resize(img, (1344, 896), cv2.INTER_CUBIC)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_input, size=1344)
    # Results
    #results.save(save_dir=save_path)  # or .show(), .save(), .crop(), .pandas(), etc.
    # images = results.render()
    # im = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
    # #cv2.imwrite(str(save_path/(str(n) + '.png')), im)
    # out.write(im)

    pred = results.pred[0].cpu().numpy()

    if pred.shape[0] > 0:
        img_mic_human = cv2.rectangle(img_human, 
                            pt1=(int(pred[0, 0]*x_ratio), int(pred[0, 1]*y_ratio)), 
                            pt2=(int(pred[0, 2]*x_ratio), int(pred[0, 3]*y_ratio)), 
                            color=(0, 0, 255), 
                            thickness=10)
        img_mic_human = cv2.putText(img_mic_human, 'Mic', (int(pred[0, 0]*x_ratio), int(pred[0, 1]*y_ratio)), font,  
                fontScale, color, thickness, cv2.LINE_AA) 
        #previsous_pts.append(twod_box)
    else:
        img_mic_human = img_human            
    out.write(img_mic_human)

    if n>200:
        break
# vis.destroy_window()
cv2.destroyAllWindows()
out.release()