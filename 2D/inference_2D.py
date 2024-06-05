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

print(sys.path)

ORI_RESO = True
# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path='/home/Tower_crane_perception/2D/runs/train/exp3/weights/last.pt', source='local')
model.iou = 0.2
model.conf = 0.01
# Images
video_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/hikrobot/")
lidar_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/livox/")
image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
lidar_list = sorted(lidar_path.rglob("*.pcd"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
save_path = Path("/home/Tower_crane_perception/2D/runs/inference/2d_lidar_livox/")
save_path.mkdir(exist_ok=True, parents=True)

if ORI_RESO:
    size = (5472, 3648)
else:
    size = (1344, 768)
out = cv2.VideoWriter(str(save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True) 


x_ratio = 5472 / 1344
y_ratio = 3648 / 768
last_pred = None
start_tracking = False
end_tracking = False
last_detected = False
not_detected_cnt = 0
not_detected_THRE = 20

previsous_pts = []
threed_boxes = []
for n, (i_p, l_p) in enumerate(zip(image_list[65:], lidar_list[65:])):
    print(i_p, l_p)
    img = cv2.imread(str(i_p))
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
        last_pred = pred
        start_tracking = True
        last_detected = True
    else:
        if start_tracking:
            pred = last_pred
        if not last_detected:
            not_detected_cnt += 1
        else:
            not_detected_cnt = 0

        if not_detected_cnt > not_detected_THRE:
            if start_tracking:
                end_tracking = True
        last_detected = False
    #print(start_tracking, end_tracking)
    if start_tracking:
        if not end_tracking:
            #print(pred)
            # if len(previsous_pts)>3:
            #     twod_box = tracking_diff(previsous_pts, 3)
            # twod_box = []
            if ORI_RESO:
                pixel_pt = ((pred[0, 0]+pred[0, 2])*x_ratio/2, (pred[0, 1]+pred[0, 3])*x_ratio/2, 1)
                print(pixel_pt, pred[0, 4])
                camera_pt = pixel2Camera(pixel_pt, -20.0)
                lidar_pt = camera2Lidar(camera_pt)
                print(pixel_pt, camera_pt, lidar_pt)
                pcd, all_cluster_centers, all_cluster_aligments = pcd_clustering(str(l_p))
                #all_cluster_centers = [[32, 6.56, -0.7]]
                print(all_cluster_centers)



                # #print(all_cluster_centers)
                # # all_cluster_centers = [[100, 40, 50],
                # #                        [200, -50, 60],
                # #                        [-40, 60, 80]]
                ind = find_closest_cluster(vecs=all_cluster_centers, vec1=lidar_pt[:3])
                threed_box = get_3d_box_from_points(all_cluster_aligments[all_cluster_aligments[:, 3]==ind])
                threed_boxes.append(threed_box)

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
            else:
                img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
                img = cv2.rectangle(img, 
                                    pt1=(int(pred[0, 0]), int(pred[0, 1])), 
                                    pt2=(int(pred[0, 2]), int(pred[0, 3])), 
                                    color=(0, 0, 255), 
                                    thickness=5)
            #previsous_pts.append(twod_box)
                
    if not start_tracking or end_tracking:
        threed_boxes.append([0,0,0,0,0,0])
        if not ORI_RESO:
            img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    
    out.write(img)

    if n>200:
        break
# vis.destroy_window()
cv2.destroyAllWindows()
out.release()
np.savetxt(save_path/'threed_boxes.txt', np.array(threed_boxes), fmt='%1.4e')