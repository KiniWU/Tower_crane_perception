import numpy as np
import cv2




img = cv2.imread("/home/tower_crane_data/gen_dataset/333-v3/train/images/hik_1_png.rf.8dd31f9e138b2f105b88d3985b53c66e.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
bright_pixel = np.amax(v)
print(bright_pixel)