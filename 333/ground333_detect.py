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

def test(model_path,video_path,save_path):
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

    hook_pos_history  = np.array([[0,0]])
    mic_pos_history   = np.array([[0,0]])
    frame_pos_history = np.array([[0,0]])
    hook_pred_history = np.array([[0,0,0,0,0,0]])
    mic_pred_history  = np.array([[0,0,0,0,0,0]])
    frame_pred_history= np.array([[0,0,0,0,0,0]])
    pred              = np.array([[0,0,0,0,0,0],[0,0,0,0,0,1]])
    is_lift_start     = False

    for n, i_p in enumerate(image_list):
        # print(i_p)
        img  = cv2.imread(str(i_p))
        
        # pred filtering
        pred = \
            g3u.pred_filtering(hook_pred_history,mic_pred_history,frame_pred_history,pred)

        # detection
        pred = g3u.ground333_detect_new(model, img, 1280)
        print("pred\n",pred)
        # print("hook_pos_history\n",hook_pos_history[n,:])

        # check mic lifting 
        hook_pred_history,mic_pred_history,frame_pred_history,is_lift_start = \
            g3u.check_mic_lifting(hook_pred_history,mic_pred_history,frame_pred_history,pred,is_lift_start)

        #plotting 
        g3u.plotting(img, pred,is_lift_start)
            
        # cv2.imwrite(save_path / (str(n) + ".png"), img)
        out.write(img)

        if n>200:
            break
    print("hook_pos_history\n", hook_pos_history)  
    # vis.destroy_window()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    model_path = Path('/home/Tower_crane_perception/333/runs/train/333_v3/weights/last.pt')
    # video_path = Path("/home/tower_crane_data/dataset_333/2024-06-12-10-55-10_luomazhou/pic")
    video_path = Path("/home/tower_crane_data/dataset_333/2024-06-14-09-33-24_ruian/pic")
    save_path  = Path("runs/detect")
    test(model_path,video_path,save_path)
