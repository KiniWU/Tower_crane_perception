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
def check_lift_start(pred_history,frequency=5,interval=5):
    frame_num = pred_history.shape[0]
    if frame_num >= frequency*interval:
        current_pixel_pos   = pred2pos(pred_history[-1,:])
        prev_pixel_pos      = pred2pos(pred_history[-frequency*interval,:])
        pixel_dist = np.linalg.norm(current_pixel_pos-prev_pixel_pos)
        if pixel_dist > 30:
            return True
        else:
            return False
    else:
        return False

def obj_multi_filter(obj_pred_history,pred,object_class):
    #deal multiple detections of one object in current frame
    object_classes = pred[:,5]
    # print("object_class\n",object_class)
    obj_index  = np.where( object_classes == object_class)
    obj_index  = obj_index[0]
    if len(obj_index) >= 2:
        print("obj_index\n",obj_index)
        center_dist = []
        for index in obj_index:
            print("obj_pred\n", pred[index,:])
            # center_x = (pred[index, 0] + pred[index, 2])/2
            # center_y = (pred[index, 1] + pred[index, 3])/2
            # current_center = np.array([[center_x,center_y]])
            current_center = pred2pos(pred[index,:])
            last_center    = pred2pos(obj_pred_history[-1,:])
            center_dist.append(np.linalg.norm(current_center-last_center))
        
        min_dist  = min(center_dist)
        min_index = center_dist.index(min_dist)
        print("min_index",min_index)
        for index in obj_index:
            if index != obj_index[min_index]:
                print("index",index)
                pred = np.delete(pred, index, axis=0)

    return pred

def obj_loss_filter(obj_pred_history,pred,object_class):
    #deal with loss of detection in current frame
    obj_loss_flag = True 
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == object_class:
            obj_loss_flag = False

    if obj_loss_flag == True:
        pred = np.vstack((pred,obj_pred_history[-1,:]))

    return pred

def obj_pred_history(obj_pred_history,pred,obj_class):
    obj_loss_flag = True
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == obj_class:
            obj_loss_flag = False
            obj_pred_history = np.vstack((obj_pred_history,pred[i,:]))

    if obj_loss_flag == True:
        # obj_pos_history = np.vstack((obj_pos_history,[[-1,-1]]))
        obj_pred_history = np.vstack((obj_pred_history,obj_pred_history[-1,:]))

    return obj_pred_history

def obj_pos_history(obj_pos_history,pred,obj_class):
    obj_loss_flag = True
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == obj_class:
            obj_loss_flag = False
            center   = pred2pos(pred[i,:])
            # print(center)
            # hook_pos_history[n,:] = center 
            obj_pos_history = np.vstack((obj_pos_history,center))

    if obj_loss_flag == True:
        # obj_pos_history = np.vstack((obj_pos_history,[[-1,-1]]))
        obj_pos_history = np.vstack((obj_pos_history,obj_pos_history[-1,:]))

    return obj_pos_history

def pred2pos(pred):
    center_x = (pred[0] + pred[2])/2
    center_y = (pred[1] + pred[3])/2
    pos      = np.array([[center_x,center_y]])
    return pos

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

hook_pos_history  = np.array([0,0])
mic_pos_history   = np.array([0,0])
frame_pos_history = np.array([0,0])
hook_pred_history = np.array([0,0,0,0,0,0])
mic_pred_history  = np.array([0,0,0,0,0,0])
frame_pred_history= np.array([0,0,0,0,0,0])
is_lift_start     = False

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
    pred = obj_multi_filter(hook_pred_history,pred,0)
    pred = obj_multi_filter(mic_pred_history,pred,1)
    pred = obj_multi_filter(frame_pred_history,pred,2)
    
    pred = obj_loss_filter(hook_pred_history,pred,0)
    pred = obj_loss_filter(mic_pred_history,pred,1)
    pred = obj_loss_filter(frame_pred_history,pred,2)
    # hook_pos_history  = obj_pos_history(hook_pos_history,pred,0) #0-hook;1-mic;2-frame;3-people
    # mic_pos_history   = obj_pos_history(mic_pos_history,pred,1) #0-hook;1-mic;2-frame;3-people
    # frame_pos_history = obj_pos_history(frame_pos_history,pred,2) #0-hook;1-mic;2-frame;3-people
    hook_pred_history  = obj_pred_history(hook_pred_history,pred,0) #0-hook;1-mic;2-frame;3-people
    mic_pred_history   = obj_pred_history(mic_pred_history,pred,1) #0-hook;1-mic;2-frame;3-people
    frame_pred_history = obj_pred_history(frame_pred_history,pred,2) #0-hook;1-mic;2-frame;3-people

    is_hook_start     = check_lift_start(hook_pred_history)
    is_mic_start      = check_lift_start(mic_pred_history)
    is_frame_start    = check_lift_start(frame_pred_history)

    if (is_hook_start and is_mic_start) or (is_hook_start and is_frame_start) or (is_mic_start and is_frame_start):
        is_lift_start = True
        # print(is_lift_start)

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
print("hook_pos_history\n", hook_pos_history)  
# vis.destroy_window()
out.release()
cv2.destroyAllWindows()
