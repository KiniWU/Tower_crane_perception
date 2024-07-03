import numpy as np
import cv2




img = cv2.imread("/home/tower_crane_data/dataset_333/2024-06-14-09-33-24_ruian/pic/hik_1.png")
print(img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)

intensity_factor = 1.7

increased_value_channel = np.clip(v * intensity_factor, 0, 255).astype(np.uint8)
img_hsv[:, :, 2] = increased_value_channel

enhanced_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite("res/image_enhanced.png", enhanced_image)