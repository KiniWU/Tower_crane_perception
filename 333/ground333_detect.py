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
import ground333_utils as g3u
# from tracking import tracking_diff
# from config   import model_path, video_path, lidar_path, save_path, img_size

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

print(sys.path)

model_path = Path('/home/Tower_crane_perception/333/runs/train/333_v1/weights/last.pt')
video_path = Path("/home/tower_crane_data/dataset_333/2024-06-12-10-55-10_luomazhou/pic")
save_path  = Path("runs/detect")
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


save_path.mkdir(exist_ok=True, parents=True)
img_size = (5472, 3648)
out = cv2.VideoWriter(str(save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, img_size, True)

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True) 


# x_ratio = img_size[0] / 1280
# y_ratio = img_size[1] / 1280

hook_pos_history  = np.array([0,0])
mic_pos_history   = np.array([0,0])
frame_pos_history = np.array([0,0])
hook_pred_history = np.array([0,0,0,0,0,0])
mic_pred_history  = np.array([0,0,0,0,0,0])
frame_pred_history= np.array([0,0,0,0,0,0])
is_lift_start     = False

for n, i_p in enumerate(image_list):
    # print(i_p)
    img  = cv2.imread(str(i_p))
    img_size = img.shape
    print(img_size)
    pred,img = g3u.ground333_detect_new(model, img, 1280)

    print("pred\n",pred)
    # print("hook_pos_history\n",hook_pos_history[n,:])

    # check moving of hook/mic_frame/mic
    # pred = g3u.obj_multi_filter(hook_pred_history, pred,0)
    # pred = g3u.obj_multi_filter(mic_pred_history,  pred,1)
    # pred = g3u.obj_multi_filter(frame_pred_history,pred,2)
    
    # pred = g3u.obj_loss_filter(hook_pred_history, pred,0)
    # pred = g3u.obj_loss_filter(mic_pred_history,  pred,1)
    # pred = g3u.obj_loss_filter(frame_pred_history,pred,2)

    # hook_pred_history  = g3u.obj_pred_history(hook_pred_history, pred,0) #0-hook;1-mic;2-frame;3-people
    # mic_pred_history   = g3u.obj_pred_history(mic_pred_history,  pred,1) #0-hook;1-mic;2-frame;3-people
    # frame_pred_history = g3u.obj_pred_history(frame_pred_history,pred,2) #0-hook;1-mic;2-frame;3-people

    # is_hook_start     = g3u.check_lift_start(hook_pred_history)
    # is_mic_start      = g3u.check_lift_start(mic_pred_history)
    # is_frame_start    = g3u.check_lift_start(frame_pred_history)

    # if (is_hook_start and is_mic_start) or (is_hook_start and is_frame_start) or (is_mic_start and is_frame_start):
    #     is_lift_start = True
        # print(is_lift_start)

    #plotting 
    for i in np.arange(pred.shape[0]):
        colors     = [[0,0,0],(0, 255, 0),(0,0,255),(255, 0, 0)]
        img  = cv2.rectangle(img, 
                    pt1=(int(pred[i, 0]), int(pred[i, 1])), 
                    pt2=(int(pred[i, 2]), int(pred[i, 3])), 
                    color=colors[int(pred[i,5])], 
                    thickness=5)
    if is_lift_start == True:
        cv2.putText(img,'the mic lifting start:',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        
    # img = cv2.resize(img, (5472, 3648), cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path / (str(n) + ".png"), img)
    out.write(img)

    if n>200:
        break
print("hook_pos_history\n", hook_pos_history)  
# vis.destroy_window()
out.release()
cv2.destroyAllWindows()
