import torch
import cv2
from pathlib import Path
from tower_utils import pixel2Camera, \
        camera2Lidar, find_closest_cluster_angle, \
        find_closest_cluster_eucli, get_3d_box_from_points, \
        lidar2Camera, camera2Pixel, draw_3d_box
import open3d as o3d
import time
import numpy as np

from tracking import tracking_diff
from config import model_path, video_path, lidar_path, save_path, img_size

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

print(sys.path)


# Load the PCD file
def check_lift_start(pos_history,frequency=5,interval=5):
    frame_num = pos_history.shape[0]
    if frame_num >= frequency*interval:
        pixel_dist = np.linalg.norm(pos_history[-1,:]-pos_history[-frequency*interval,:])
        if pixel_dist > 30:
            return True
        else:
            return False
    else:
        return False



ORI_RESO = True


# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.iou = 0.2
model.conf = 0.5
# Images
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]) \
                    if is_int(str(a).split("_")[-1].split(".")[0]) else float('inf'))

save_path  = Path("runs/detect")
save_path.mkdir(exist_ok=True, parents=True)

if ORI_RESO:
    size = img_size
else:
    size = (1280, 1280)
out = cv2.VideoWriter(str(save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True) 


x_ratio = img_size[0] / 1280
y_ratio = img_size[1] / 1280


previsous_pts = []
threed_boxes  = []
# hook_pos_history   = np.full([200,2],-1)
hook_pos_history = np.array([0,0])
mic_box    = np.array([])
frame_box  = np.array([])


for n, i_p in enumerate(image_list):
    # print(i_p)
    img       = cv2.imread(str(i_p))
    img_input = cv2.resize(img, (1280, 1280), cv2.INTER_CUBIC)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_input, size=1280)
    # Results

    pred = results.pred[0].cpu().numpy()

    print("pred\n",pred)
    # print("hook_pos_history\n",hook_pos_history[n,:])

    # check moving of hook/mic_frame/mic
    # index = np.where(pred[:,5] == 0)
    # index = index[0][0]
    # print(index)

    for i in np.arange(pred.shape[0]):
        if pred[i,5] == 0:
            center_x = (pred[i, 0] + pred[i, 2])/2
            center_y = (pred[i, 1] + pred[i, 3])/2
            center   = np.array([[center_x,center_y]])
            print(center)
            # hook_pos_history[n,:] = center 
            hook_pos_history = np.vstack((hook_pos_history,center))

    is_lift_start = check_lift_start(hook_pos_history)

    #plotting 
    for i in np.arange(pred.shape[0]):
        colors     = [[0,0,0],(0, 255, 0),(0,0,255),(255, 0, 0)]
        img_input  = cv2.rectangle(img_input, 
                    pt1=(int(pred[i, 0]), int(pred[i, 1])), 
                    pt2=(int(pred[i, 2]), int(pred[i, 3])), 
                    color=colors[int(pred[i,5])], 
                    thickness=5)
    if is_lift_start == True:
        cv2.putText(img_input,'the mic lifting start:',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        
    img = cv2.resize(img_input, (5472, 3648), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path / (str(n) + ".png"), img)
    out.write(img)

    if n>200:
        break
print(hook_pos_history)   
# vis.destroy_window()
out.release()
cv2.destroyAllWindows()
