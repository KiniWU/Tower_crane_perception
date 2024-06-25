#! /usr/bin/env python3

import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

import sys
sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')

import os
import time
import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
import numpy as np

from matplotlib import pyplot as plt

IMAGE_WIDTH=5472
IMAGE_HEIGHT=3648
ros_image=0

def detect(img):
    '''
    @note: 需要resize以及完成识别以及绘制的工作
    @input: img,从ROS回调image_callback中接收的图像（IMAGE_WIDTH*IMAGE_HEIGHT），
    @return: bool, 返回是否检测到目标的bool类型
    '''

    # 并在识别完成后发布这个图像
    publish_image(im1)

def tracking():
    '''
    @note: 原有inference中识别后进行的tracking工作, 追踪结果保存在同一数组或者全局的队列当中
    @input: 目标位置的数据
    @return: bool, 是否追踪成功
    '''

def image_callback(image):
    global ros_image
    ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    # print("Image shape:", ros_image.shape)  # 打印图像尺寸
    # print("Image dtype:", ros_image.dtype)  # 打印数据类型
    # print("Is image read-only?", ros_image.flags['WRITEABLE'])
    with torch.no_grad():
        detect(ros_image)
def publish_image(imgdata):
    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1241*3
    image_pub.publish(image_temp)

if __name__ == '__main__':

    rospy.init_node('ros_yolo')
    image_topic = "/hikrobot_camera/rgb"
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1, buff_size=5242880000)
    image_pub = rospy.Publisher('/yolo_result_out', Image, queue_size=1)

    rospy.spin()