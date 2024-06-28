import numpy as np
import cv2

# 定义一个点类来处理像素点
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 定义8邻域
connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]

# 区域增长函数
def region_growing(img, seeds, thresh):
    seed_list = seeds
    label = 255
    output = np.zeros_like(img)
    while(len(seed_list) > 0):
        seed_curr = seed_list.pop(0)
        output[seed_curr.y, seed_curr.x] = label
        for i in range(8):
            x_new = seed_curr.x + connects[i].x
            y_new = seed_curr.y + connects[i].y
            if x_new < 0 or y_new < 0 or x_new >= img.shape[1] or y_new >= img.shape[0]:
                continue
            if img[y_new, x_new] < thresh and output[y_new, x_new] == 0:
                output[y_new, x_new] = label
                seed_list.append(Point(x_new, y_new))
    return output

# 读取图像并转换为灰度图
img = cv2.imread('/home/haochen/HKCRC/3D_object_detection/Tower_crane_perception/data_process/camera1_300.png', cv2.IMREAD_GRAYSCALE)

# 设置种子点和阈值
seeds = [Point(10, 10)]  # 假设的种子点
thresh = 10  # 阈值

# 应用区域增长算法
result = region_growing(img, seeds, thresh)

# 显示结果
cv2.imshow('Region Growing', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
