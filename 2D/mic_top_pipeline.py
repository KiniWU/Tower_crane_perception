import torch
from config import img_size, pixel2Camera
import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent/"data_process"))
sys.path.append(str(Path(__file__).parent))

from small_human_detection_im_crop import crop_im

class App:
    def __init__(self) -> None:
        self.video_path = ""
        self.depth_path = ""
        self.mic_model_path = ""
        self.person_model_path = ""

        self.image_size = img_size

        self.model_im_size = 1280

        self.x_ratio, self.y_ratio = self.image_size / self.model_im_size
        
        pass

    def init_model(self):
        self.mic_model = torch.hub.load('yolov5', 'custom', path=self.mic_model_path, source='local')
        self.mic_model.iou = 0.2
        self.mic_model.conf = 0.7

        self.person_model = torch.hub.load('yolov5', 'custom', path=self.person_model_path, source='local')
        self.person_model.iou = 0.2
        self.person_model.conf = 0.2

    def read_video(self):
        self.image_video = sorted(self.video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
        self.depth_video = sorted(self.depth_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))

    def rotate_pt_with_imc(self, pt):
        x, y = pt
        x -= self.image_size[0] / 2
        y -= self.image_size[1] / 2

        return (self.image_size[0]/2 -x, self.image_size[1] / 2 - y)

    def depth_cluster(self, pts, method="KMeans"):
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

        Z = np.argmin(kmeans.cluster_centers_)

        return Z

    def mic_2D_detect_dep(self, img_input, dep):
        img_input = cv2.resize(img_input, (self.model_im_size, self.model_im_size), cv2.INTER_CUBIC)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        results = self.mic_model(img_input, size=self.model_im_size)

        pred = results.pred[0].cpu().numpy()

        if pred.shape[0] > 0:
            pixel_pt = np.array([[(pred[0, 0]+pred[0, 2])*self.x_ratio/2], 
                                    [(pred[0, 1]+pred[0, 3])*self.y_ratio/2]])
            pixel_pt_rot = self.rotate_pt_with_imc(pixel_pt) # x,y

            left_top = self.rotate_pt_with_imc((pred[0, 0], pred[0, 1])) # x_min, y_min --> x_max, y_max
            right_bottom = self.rotate_pt_with_imc((pred[0, 2], pred[0, 3])) # x_max, y_max --> x_min, y_min

            depth_area = dep[right_bottom[0]:left_top[0], right_bottom[1]:left_top[1]]

            depth_area_pts = np.where(depth_area>0)
            Z = self.depth_cluster(depth_area_pts)

            camera_pt = pixel2Camera(pixel_pt_rot, Z)

        return camera_pt, (right_bottom[0], right_bottom[1], left_top[0], left_top[1])
    
    def person_2D_detect_dep(self, img_input, dep):
        
        # img_input = cv2.resize(img_input, (self.model_im_size, self.model_im_size), cv2.INTER_CUBIC)
        # img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        
        img_detected = np.zeros_like(img_input, np.uint8)
        ori_w, ori_h = img_size

        num_batches = (8, 8)
        size = (648, 648)

        cropped_w = ori_w//num_batches[0]
        cropped_h = ori_h//num_batches[1]
        x_ratio = cropped_w / size[0] 
        y_ratio = cropped_h / size[1]
        img_patches = crop_im(img_input, num_batches=num_batches)
        people_bboxes = []

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
                results = self.person_model(img_input, size=size[0])

                pred = results.pred[0].cpu().numpy()
                print("yolo pred\n",pred)
                if pred.shape[0] > 0:

                    x_min = pred[0, 0]*x_ratio + x_start
                    y_min = pred[0, 2]*x_ratio + x_start
                    x_max = pred[0, 1]*y_ratio + y_start
                    y_max = pred[0, 3]*y_ratio + y_start

                    left_top = self.rotate_pt_with_imc((x_min, y_min)) # x_min, y_min --> x_max, y_max
                    right_bottom = self.rotate_pt_with_imc((x_max, y_max)) # x_max, y_max --> x_min, y_min

                    people_bboxes.append([right_bottom[0], right_bottom[1], left_top[0], left_top[1]]) # x_min, y_min, x_max, y_max, after roration
        
        if len(people_bboxes) == 0:
            return None
        else:
            return people_bboxes       


    def run(self):
        for n, (i_p, d_p) in enumerate(zip(self.image_video, self.depth_video)):
            print(i_p)
            # print(n)
            img = cv2.imread(str(i_p))
            dep = cv2.imread(str(d_p))
            img_input = cv2.rotate(img, cv2.ROTATE_180_CLOCKWISE)

            mic_pt, mic_box_2d = self.mic_2D_detect_dep(img_input, dep)
            people_bboxes = self.person_2D_detect_dep(img_input, dep)

            img_detected = img
            if mic_pt is not None:
                img_detected = cv2.rectangle(img, 
                                    pt1=(int(mic_box_2d[0]), int(mic_box_2d[1])), 
                                    pt2=(int(mic_box_2d[2]), int(mic_box_2d[3])), 
                                    color=(0, 0, 255), 
                                    thickness=5)
            if people_bboxes is not None:
                for p_b in people_bboxes:
                    img_detected = cv2.rectangle(img, 
                                        pt1=(int(p_b[0]), int(p_b[1])), 
                                        pt2=(int(p_b[2]), int(p_b[3])), 
                                        color=(0, 255, 0), 
                                        thickness=5)

            cv2.imwrite(str(n) + ".png", img_detected)
            

if __name__ == "__main__":
    app = App()
    app.run()