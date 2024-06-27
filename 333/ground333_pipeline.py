import torch
import cv2
import numpy as np
from config import Intrinsic, c2L, camera_dist

def obj_multi_filter(obj_pred_history,pred,object_class):
    # deal with multiple detections for the object(only allows one detection)
    # inputs :
    # outputs:
    object_classes = pred[:,5]
    # print("object_class\n",object_class)
    obj_index  = np.where( object_classes == object_class)
    obj_index  = obj_index[0]
    if len(obj_index) >= 2:
        print("obj_index\n",obj_index)
        center_dist = []
        for index in obj_index:
            print("obj_pred\n", pred[index,:])
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
    # deal with detection lossing for the object
    # inputs :
    # outputs:
    obj_loss_flag = True 
    for i in np.arange(pred.shape[0]):
        if pred[i,5] == object_class:
            obj_loss_flag = False

    if obj_loss_flag == True:
        # print(obj_pred_history.shape)
        pred = np.vstack((pred,obj_pred_history[-1,:]))

    return pred

def check_lift_start(pred_history,frequency=5,interval=5):
    # check whether mic lifting start 
    # inputs :
    # outputs:
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

def obj_pred_history(obj_pred_history,pred,obj_class):
    # log object prediction 
    # inputs :
    # outputs:
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


class APP333:

    def __init__(self) -> None:
        self.video_path = ""
        self.save_path  = ""
        self.model_path = ""
        self.image_size = ""
        self.model_im_size = 1280
        self.image      = ""

        self.hook_pos_history  = np.array([[0,0]])
        self.mic_pos_history   = np.array([[0,0]])
        self.frame_pos_history = np.array([[0,0]])
        self.hook_pred_history = np.array([[0,0,0,0,0,0]])
        self.mic_pred_history  = np.array([[0,0,0,0,0,0]])
        self.frame_pred_history= np.array([[0,0,0,0,0,0]])
        self.pred              = np.array([[0,0,0,0,0,0],[0,0,0,0,0,1]])
        self.is_lift_start     = False
        pass

    def init_model(self):
        self.model = torch.hub.load('yolov5', 'custom', path=self.model_path, source='local')
        self.model.iou = 0.2
        self.model.conf = 0.7


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

        # yolo inference results,pred = [xmin ymin xmax ymax confidence class name]
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

    def pred_filtering(self):
        hook_pred_history = self.hook_pred_history
        mic_pred_history  = self.mic_pred_history
        frame_pred_history= self.frame_pred_history
        pred              = self.pred

        pred = obj_multi_filter(hook_pred_history, pred,0)
        pred = obj_multi_filter(mic_pred_history,  pred,1)
        pred = obj_multi_filter(frame_pred_history,pred,2)
        
        pred = obj_loss_filter(hook_pred_history, pred,0)
        pred = obj_loss_filter(mic_pred_history,  pred,1)
        pred = obj_loss_filter(frame_pred_history,pred,2)

        self .pred = pred


    def check_mic_lifting(self):
        hook_pred_history = self.hook_pred_history
        mic_pred_history  = self.mic_pred_history
        frame_pred_history= self.frame_pred_history
        pred              = self.pred
        is_lift_start     = self.is_lift_start
    
        hook_pred_history  = obj_pred_history(hook_pred_history, pred,0) #0-hook;1-mic;2-frame;3-people
        mic_pred_history   = obj_pred_history(mic_pred_history,  pred,1) #0-hook;1-mic;2-frame;3-people
        frame_pred_history = obj_pred_history(frame_pred_history,pred,2) #0-hook;1-mic;2-frame;3-people

        is_hook_start     = check_lift_start(hook_pred_history)
        is_mic_start      = check_lift_start(mic_pred_history)
        is_frame_start    = check_lift_start(frame_pred_history)

        if (is_hook_start and is_mic_start) or (is_hook_start and is_frame_start) or (is_mic_start and is_frame_start):
            is_lift_start = True
            print(is_lift_start)

        self.hook_pred_history  = hook_pred_history
        self.mic_pred_history   = mic_pred_history
        self.frame_pred_history = frame_pred_history
        self.is_lift_start      = is_lift_start

    def plotting(self):
        img  = self.img
        pred = self.pred
        is_lift_start = self.is_lift_start
        for i in np.arange(pred.shape[0]):
            colors = [[0,0,0],(0, 255, 0),(0,0,255),(255, 0, 0)]
            img    = cv2.rectangle(img, 
                        pt1=(int(pred[i, 0]), int(pred[i, 1])), 
                        pt2=(int(pred[i, 2]), int(pred[i, 3])), 
                        color=colors[int(pred[i,5])], 
                        thickness=10)
        if is_lift_start == True:
            cv2.putText(img,'the mic lifting start:',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    def run(self):
        out = cv2.VideoWriter(str(self.save_path / "video_comp.avi"), \
                              cv2.VideoWriter_fourcc(*'MPEG'), 10, self.img_size, True)
        for n, i_p in enumerate(self.image_list):
            # print(i_p)
            img  = cv2.imread(str(i_p))
            
            # pred filtering
            self.pred_filtering()

            # detection
            self.ground333_detect()
            print("pred\n",self.pred)

            # check mic lifting 
            self.check_mic_lifting()

            #plotting 
            self.plotting()
                
            # cv2.imwrite(save_path / (str(n) + ".png"), img)
            out.write(img)

            if n>200:
                break
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app333 = APP333()
    app333.run()
