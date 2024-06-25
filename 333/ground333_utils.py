import cv2
import numpy as np
from config import Intrinsic, c2L, camera_dist


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
        pred = np.vstack((pred,obj_pred_history[-1,:]))

    return pred


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

def ground333_detect(model, img, std_img_size=1280):
    img = cv2.resize(img, (std_img_size, std_img_size), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # yolo inference results
    results = model(img, size=std_img_size)
    pred = results.pred[0].cpu().numpy()

    # img = cv2.resize(img, (5472, 3648), cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return pred,img

def ground333_tracking():
    pass
    return
