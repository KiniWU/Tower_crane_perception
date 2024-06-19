import cv2
import numpy as np
from matplotlib import pyplot as plt 

depth = cv2.imread("/home/tower_crane_data/some_test/dep/depth_0.png", cv2.IMREAD_UNCHANGED)
print(type(depth))

#depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
# print(depth.shape)
# h, w = depth.shape
# for i in range(h):
#     for j in range(w):
#         if depth[i, j] != 0:
#             print(depth[i, j])
depth = np.array(depth, np.float16)
print(depth.max())
depth /= depth.max()/255.0

depth = np.array(depth, np.uint8)
# kernel = np.ones((1000,1000),np.uint8)*255
# cv2.dilate(depth, kernel,iterations = 10)
cv2.imwrite("depth.png", depth)

hist = cv2.calcHist([depth],[0],None,[256],[0,256]) 
  
# plot the above computed histogram 
plt.plot(hist, color='b') 
plt.title('Image Histogram For Blue Channel GFG') 
plt.savefig("hist.png")