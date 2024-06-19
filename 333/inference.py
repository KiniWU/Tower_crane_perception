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

from pointCloud_clustering import pcd_clustering

print(sys.path)

ORI_RESO = True


# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.iou = 0.2
model.conf = 0.7
# Images
image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
lidar_list = sorted(lidar_path.rglob("*.pcd"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
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
threed_boxes = []
for n, (i_p, l_p) in enumerate(zip(image_list, lidar_list)):
    print(i_p, l_p)
    img = cv2.imread(str(i_p))
    img_input = cv2.resize(img, (1280, 786), cv2.INTER_CUBIC)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_input, size=1344)
    # Results

    pred = results.pred[0].cpu().numpy()

    pixel_pt = np.array([[(pred[0, 0]+pred[0, 2])*x_ratio/2], 
                            [(pred[0, 1]+pred[0, 3])*x_ratio/2]])
    # print(pixel_pt, pred[0, 4])
    # camera_pt = pixel2Camera(pixel_pt, -20.0)
    # lidar_pt = camera2Lidar(camera_pt)
    # print(pixel_pt, camera_pt, lidar_pt)
    pcd, all_cluster_centers, all_cluster_aligments = pcd_clustering(str(l_p))
    #all_cluster_centers = [[32, 6.56, -0.7]]
    print(all_cluster_centers)

    lidar_centers = np.ones((4, len(all_cluster_centers)), np.float32)
    lidar_centers[:3, :] = np.transpose(np.array(all_cluster_centers))

    camera_centers = lidar2Camera(lidar_centers)
    pixel_centers = camera2Pixel(camera_centers)
    ind = find_closest_cluster_eucli(vecs=pixel_centers[:2, :], vec1=pixel_pt)
    threed_box = get_3d_box_from_points(all_cluster_aligments[all_cluster_aligments[:, 3]==ind])
    threed_boxes.append(threed_box)

    img = draw_3d_box(img=img, threed_box=threed_box, line_color_c=(0, 0, 255), line_thickness=10)
    img = cv2.resize(img, (1344, 893), cv2.INTER_CUBIC)
    cv2.imwrite(save_path / (str(n) + ".png"), img)

    
    out.write(img)

    if n>200:
        break
# vis.destroy_window()
out.release()
np.savetxt(save_path/'threed_boxes.txt', np.array(threed_boxes), fmt='%1.4e')
cv2.destroyAllWindows()
