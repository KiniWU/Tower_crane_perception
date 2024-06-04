import cv2
import torch
import cv2
from pathlib import Path
from tower_utils import pixel2Camera, camera2Lidar, find_closest_cluster, get_3d_box_from_points
import open3d as o3d
import time
import numpy as np

from tracking import tracking_diff

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

from pointCloud_clustering import pcd_clustering


x_ratio = 3840 / 1344
y_ratio = 2160 / 768

def get_3d_box(img, pcd, model):
    img_input = cv2.resize(img, (1344, 768), cv2.INTER_CUBIC)
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
        pixel_pt = ((pred[0, 0]+pred[0, 2])*x_ratio/2, (pred[0, 1]+pred[0, 3])*x_ratio/2, 1)
        print(pixel_pt, pred[0, 4])
        camera_pt = pixel2Camera(pixel_pt, 50.0)
        lidar_pt = camera2Lidar(camera_pt)
        print(pixel_pt, camera_pt, lidar_pt)
        pcd, all_cluster_centers, all_cluster_aligments = pcd_clustering(pcd)
        print(all_cluster_centers)


        # #print(all_cluster_centers)
        # # all_cluster_centers = [[100, 40, 50],
        # #                        [200, -50, 60],
        # #                        [-40, 60, 80]]
        ind = find_closest_cluster(vecs=all_cluster_centers, vec1=lidar_pt[:3])
        threed_box = get_3d_box_from_points(all_cluster_aligments[all_cluster_aligments[:, 3]==ind])

        # vis.add_geometry(pcd)
        # vis.run()
        # # vis.update_geometry(pcd)
        # # vis.poll_events()
        # # vis.update_renderer()
        # time.sleep(1)
        # vis.capture_screen_image(str(save_path / (str(n) + ".png")))
        # print(all_cluster_centers[ind], threed_box)
        # tracking_diff()
        img = cv2.rectangle(img, 
                            pt1=(int(pred[0, 0]*x_ratio), int(pred[0, 1]*y_ratio)), 
                            pt2=(int(pred[0, 2]*x_ratio), int(pred[0, 3]*y_ratio)), 
                            color=(0, 0, 255), 
                            thickness=10)
        return img
    else:
        return img

def app():

    # read picture and lidar

    # 
    model = torch.hub.load('yolov5', 'custom', path='/home/Tower_crane_perception/2D/runs/train/exp2/weights/last.pt', source='local')
    model.iou = 0.2
    model.conf = 0.7

    
    cap = cv2.VideoCapture("rtsp://root:pass@192.168.0.91:554/axis-media/media.amp")

    while(cap.isOpened()):
        ret, frame = cap.read()
        detected_img = get_3d_box(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app()