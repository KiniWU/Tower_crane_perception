import torch
import cv2
import numpy as np
from config import Intrinsic, c2L, camera_dist
from pathlib import Path
from sklearn.cluster import KMeans, MeanShift

def tracking_diff(previous_pts, back = 3):
    previous_pts = np.array(previous_pts)
    diff_x, diff_y = np.mean(previous_pts[-1*back:-1, :] - previous_pts[-1*(back+1):-2, :], dim=1)

    return (previous_pts[-1, 0] + diff_x, previous_pts[-1, 1] + diff_y)


def kalman_filter_init(initial_state):
    # init kalman filter
    Delta_t = 0.2
    kalman_filter = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements 
    kalman_filter.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix    = np.array([[1, 0, Delta_t, 0], [0, 1, 0, Delta_t], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman_filter.processNoiseCov     = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0],[0, 0, 0, 10]], np.float32) 
    kalman_filter.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
    kalman_filter.statePre            = initial_state
    kalman_filter.statePost           = initial_state
    return kalman_filter

def obj_kalman_filter(kalman_filter,measured_state):
    if measured_state.shape[0] != 0:
        # correction
        kalman_filter.correct(measured_state)
    # prediction
    predicted_state = kalman_filter.predict()

    return predicted_state

def obj_simple_filter(obj_pred_history,pred,object_class):
    # deal with detection lossing for the object
    # inputs :
    # outputs:
    obj_loss_flag = True 
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == object_class:
            obj_loss_flag = False

    if obj_loss_flag == True:
        # print("obj_pred_history[-1]\n",obj_pred_history[-1])
        pred = np.vstack((pred,obj_pred_history[-1]))

    return pred

def check_obj_lift(pred_history,frequency=5,interval=5):
    # check whether mic lifting start 
    # inputs :
    # outputs:

    frame_num = len(pred_history)
    if frame_num >= frequency*interval:
        current_pixel_pos   = pred2pos(pred_history[-1])
        prev_pixel_pos      = pred2pos(pred_history[-frequency*interval])
        pixel_dist = np.linalg.norm(current_pixel_pos-prev_pixel_pos)
        if pixel_dist > 100:
            return True
        else:
            return False
    else:
        return False

def obj_pred_history(obj_pred_history,pred,obj_class):
    # log object prediction 
    # inputs :
    # outputs:
    obj_loss_flag = True
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == obj_class:
            obj_loss_flag = False
            obj_pred_history.append(pred[i,:].tolist())
            # obj_pred_history = np.vstack((obj_pred_history,pred[i,:]))

    if obj_loss_flag == True:
        # obj_pos_history = np.vstack((obj_pos_history,[[-1,-1]]))
        # obj_pred_history = np.vstack((obj_pred_history,obj_pred_history[-1,:]))
        obj_pred_history.append(obj_pred_history[-1])

    return obj_pred_history

def obj_pos_history(obj_pos_history,pred,obj_class):
    # log object pixel center position 
    # inputs :
    # outputs:
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
    # convert object prediction to object center pixel position 
    # inputs :
    # outputs:
    center_x = (pred[0] + pred[2])/2
    center_y = (pred[1] + pred[3])/2
    pos      = np.array([[center_x,center_y]])
    return pos

def pos2pred(pos,current_pred,latest_pred):
    # convert object prediction to object center pixel position 
    # inputs :
    # outputs:
    if current_pred.shape[0] != 0:
        pred = current_pred
    else:
        pred = latest_pred

    box_xlength = pred[0,2] - pred[0,0]
    box_ylength = pred[0,3] - pred[0,1]
    center_x = pos[0,0]
    center_y = pos[0,1]
    xmin = center_x - box_xlength/2
    xmax = center_x + box_xlength/2
    ymin = center_y - box_ylength/2
    ymax = center_y + box_ylength/2
    pred_ = np.array([[xmin,ymin,xmax,ymax,pred[0,4],pred[0,5]]])
    return pred_


class APP333:

    def __init__(self) -> None:
        # self.image_path = Path("/home/tower_crane_data/dataset_333/2024-06-14-09-33-24_ruian/pic")
        self.image_path = Path("/home/tower_crane_data/dataset_333/2024-06-12-10-55-10_luomazhou/pic")
        self.depth_path = Path("/home/tower_crane_data/dataset_333/2024-06-12-10-55-10_luomazhou/dep")
        self.model_path = Path('/home/Tower_crane_perception/333/runs/train/333_v1/weights/best.pt')
        self.save_path  = Path("runs/detect")
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.image      = ""
        self.dep        = ""
        self.image_size = (5472, 3648)
        self.model_im_size = 1280
        # kalman filter
        self.is_micKF_init = False
        
        #
        self.hook_pred   = np.empty((0,6))
        self.mic_pred    = np.empty((0,6))
        self.frame_pred  = np.empty((0,6))
        self.people_pred = np.empty((0,6))

        self.mic_latest_pred    = np.empty((0,6))

        self.hook_pred_history = [[0,0,0,0,0,0]]
        self.mic_pred_history  = [[0,0,0,0,0,1]]
        self.frame_pred_history= [[0,0,0,0,0,2]]
        self.pred              = np.array([[0,0,0,0,0,0],[0,0,0,0,0,1]])
        self.filtered_pred     = np.empty((0,6))
        self.is_lift_start     = False
        pass
    
    def read_model(self):
        # read model
        self.model = torch.hub.load('yolov5', 'custom', path=self.model_path, source='local')
        self.model.iou = 0.2
        self.model.conf = 0.5

        self.kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        self.mean_shift = MeanShift()


    def read_video(self):
        self.image_list = sorted(self.image_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
        self.depth_list = sorted(self.depth_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))

    def ground333_detect(self):
        std_img_size = self.model_im_size
        model        = self.model
        img          = self.image

        img_size = img.shape
        IMAGE_WIDTH  = img_size[1]
        IMAGE_HEIGHT = img_size[0]
        WIDTH_RATIO  = IMAGE_WIDTH/std_img_size
        HEIGHT_RATIO = IMAGE_HEIGHT/std_img_size
        #convert image to yolo model required format
        img = cv2.resize(img, (std_img_size, std_img_size), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # yolo inference results,pred = [xmin ymin xmax ymax confidence class]
        results = model(img, size=std_img_size)
        pred = results.pred[0].cpu().numpy()
        # convert yolo prediction from yolo reqired img size to raw image size
        pred[:,0] = pred[:,0]*WIDTH_RATIO
        pred[:,2] = pred[:,2]*WIDTH_RATIO
        pred[:,1] = pred[:,1]*HEIGHT_RATIO
        pred[:,3] = pred[:,3]*HEIGHT_RATIO

        # convert image to original format after yolo reference 
        # img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.pred = pred

    def pred_classification(self):

        pred = self.pred
        self.hook_pred   = np.empty((0,6))
        self.mic_pred    = np.empty((0,6))
        self.frame_pred  = np.empty((0,6))
        self.people_pred = np.empty((0,6))

        for pred_i in pred:
            if pred_i[5]   == 0:
                self.hook_pred = np.vstack((self.hook_pred,pred_i))
            elif pred_i[5] == 1:
                self.mic_pred = np.vstack((self.mic_pred,pred_i))
            elif pred_i[5] == 2:
                self.frame_pred = np.vstack((self.frame_pred,pred_i))
            elif pred_i[5] == 3:
                self.people_pred = np.vstack((self.people_pred,pred_i))
        
        if self.hook_pred.shape[0]>1:
            sorted_indices = np.argsort(self.hook_pred[:, 4])[::-1]
            self.hook_pred = self.hook_pred[sorted_indices]

        if self.mic_pred.shape[0]>1:
            sorted_indices = np.argsort(self.mic_pred[:, 4])[::-1]
            self.mic_pred = self.mic_pred[sorted_indices]

        if self.frame_pred.shape[0]>1:
            sorted_indices = np.argsort(self.frame_pred[:, 4])[::-1]
            self.frame_pred = self.frame_pred[sorted_indices]

        if self.people_pred.shape[0]>1:
            sorted_indices = np.argsort(self.people_pred[:, 4])[::-1]
            self.people_pred = self.people_pred[sorted_indices]

        # update obj latest pred
        if self.mic_pred.shape[0]>0:
            self.mic_latest_pred = self.mic_pred

    def obj_multi_filter(self):
        if self.hook_pred.shape[0]>1:
            self.hook_pred = self.hook_pred[0,:] # keep the prediction with highest probability
            self.hook_pred = self.hook_pred.reshape((1,6)) # convert 1D array to 2D array

        if self.mic_pred.shape[0]>1:
            self.mic_pred = self.mic_pred[0,:]
            self.mic_pred = self.mic_pred.reshape((1,6))

        if self.frame_pred.shape[0]>1:
            self.frame_pred = self.frame_pred[0,:]
            self.frame_pred = self.frame_pred.reshape((1,6))
    
    def obj_loss_filter(self): 
        # init the mic kalman filter
        if self.is_micKF_init == False:
            if self.mic_pred.shape[0] != 0:
                self.is_micKF_init = True
                center_x    = (self.mic_pred[0,0] + self.mic_pred[0,2])/2
                center_y    = (self.mic_pred[0,1] + self.mic_pred[0,3])/2
                init_state  = np.array([center_x, center_y, 0, 0], np.float32)
                self.mic_KF = kalman_filter_init(init_state)
        # propogate the mic kalman filter
        if  self.is_micKF_init == True:
            if self.mic_pred.shape[0] != 0:
                center_x    = (self.mic_pred[0,0] + self.mic_pred[0,2])/2
                center_y    = (self.mic_pred[0,1] + self.mic_pred[0,3])/2
                measured_state = np.array([ [center_x], [center_y]], dtype=np.float32)
            else:
                measured_state = np.array([])

            # print("measured_state\n",measured_state)
            corrected_state = obj_kalman_filter(self.mic_KF, measured_state)
            corrected_state = corrected_state.reshape(1,4) #convert to 1D array to 2D array
            # print("corrected_state\n",corrected_state)
            self.mic_pred = pos2pred(corrected_state,self.mic_pred, self.mic_latest_pred)
            # print("self.mic_pred\n",self.mic_pred)

        # print("pred\n",pred)
        # self.history_log()
        # self.pred = obj_simple_filter(self.hook_pred_history, self.pred,0)
        # self.pred = obj_simple_filter(self.mic_pred_history,  self.pred,1)
        # self.pred = obj_simple_filter(self.frame_pred_history,self.pred,2)
        # print("filered pred\n",pred)

    def history_log(self):
        hook_pred_history = self.hook_pred_history
        mic_pred_history  = self.mic_pred_history
        frame_pred_history= self.frame_pred_history
        pred              = self.pred
        if len(hook_pred_history) <=50:
            hook_pred_history  = obj_pred_history(hook_pred_history, pred,0) #0-hook;1-mic;2-frame;3-people
        else:
            hook_pred_history.pop(0)
        
        if len(mic_pred_history) <=50:
            mic_pred_history   = obj_pred_history(mic_pred_history,  pred,1) #0-hook;1-mic;2-frame;3-people
        else:
            mic_pred_history.pop(0)

        if len(frame_pred_history) <=50:
            frame_pred_history = obj_pred_history(frame_pred_history,pred,2) #0-hook;1-mic;2-frame;3-people
        else:
            frame_pred_history.pop(0)
        
        self.hook_pred_history = hook_pred_history
        self.mic_pred_history  = mic_pred_history
        self.frame_pred_history= frame_pred_history

    def check_mic_lifting_2D(self):
        hook_pred_history = self.hook_pred_history
        mic_pred_history  = self.mic_pred_history
        frame_pred_history= self.frame_pred_history
        pred              = self.pred
    
        is_hook_start     = check_obj_lift(hook_pred_history)
        is_mic_start      = check_obj_lift(mic_pred_history)
        is_frame_start    = check_obj_lift(frame_pred_history)

        if (is_hook_start and is_mic_start) or (is_hook_start and is_frame_start) or (is_mic_start and is_frame_start):
            self.is_lift_start = True
            print(self.is_lift_start)

    def check_mic_lifting(self):
        mic_pred = self.mic_pred
        dep = self.dep
        depth_area = dep[int(mic_pred[0,0]):int(mic_pred[0,2]), int(mic_pred[0,1]):int(mic_pred[0,3])]
        # print("depth_area\n",depth_area)
        depth_info = depth_area[np.where(depth_area>0)]
        print("depth_info\n",depth_info)
        Z, mic_coor = self.depth_cluster(np.array(depth_info, np.float32).reshape(-1, 1)) # h,w, i.e., y, x
        pass

    def depth_cluster(self, pts, method="KMeans"):
        kmeans_res = self.kmeans.fit(pts)

        Z = np.min(kmeans_res.cluster_centers_)

        label_min = np.argmin(kmeans_res.cluster_centers_)

        #print(kmeans_res.cluster_centers_)

        return Z, np.where(kmeans_res.labels_==label_min)

    def plotting(self):
        img  = self.image
        pred = self.pred
        is_lift_start = self.is_lift_start
        self.filtered_pred = np.vstack((self.hook_pred,\
                            self.mic_pred,self.frame_pred,self.people_pred))
        # print("self.filtered_pred\n",self.filtered_pred)
        for i in np.arange(self.filtered_pred.shape[0]):
            colors = [[0,0,0],(0, 255, 0),(0,0,255),(255, 0, 0)]
            img    = cv2.rectangle(img, 
                        pt1=(int(self.filtered_pred[i, 0]), int(self.filtered_pred[i, 1])), 
                        pt2=(int(self.filtered_pred[i, 2]), int(self.filtered_pred[i, 3])), 
                        color=colors[int(self.filtered_pred[i,5])], 
                        thickness=10)
        if is_lift_start == True:
            cv2.putText(img,'the mic lifting start:',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),10)

    def run(self):
        self.read_model()
        self.read_video()
        out = cv2.VideoWriter(str(self.save_path / "video_comp.avi"), \
                              cv2.VideoWriter_fourcc(*'MPEG'), 10, self.image_size, True)
        for n, (i_p,d_p) in enumerate(zip(self.image_list,self.depth_list)):
            # print(i_p)
            self.image  = cv2.imread(str(i_p))
            self.dep    = cv2.imread(str(d_p), cv2.IMREAD_UNCHANGED)
            # self.dep = self.dep[np.where(self.dep>0)]
            # print("dep\n",self.dep)
            # print(self.dep)
            # detection
            self.ground333_detect()
            print("pred\n",self.pred)
            # print("filtered pred\n",self.pred)

            self.pred_classification()
            # print("mic_pred\n",self.mic_pred)
            
            self.obj_multi_filter()
            print("mic_pred\n",self.mic_pred)
            
            # pred filtering
            self.obj_loss_filter()

            # check mic lifting 
            self.check_mic_lifting()

            #plotting 
            self.plotting()
                
            # cv2.imwrite(save_path / (str(n) + ".png"), img)
            out.write(self.image)

            if n>200:
                break

        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app333 = APP333()
    app333.run()
