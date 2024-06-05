import cv2
import numpy as np

# 初始化 Kalman Filter
kalman_filter = cv2.KalmanFilter(4, 2)  # 4维状态，2维测量
kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4

# 假设你从点云中获取了目标框的测量值（例如，目标框的中心位置）
measured_state = np.array([[measured_x], [measured_y]], np.float32)

# 预测步骤
predicted_state = kalman_filter.predict()

# 更新步骤
corrected_state = kalman_filter.correct(measured_state)

# 获取更新后的目标框位置
corrected_x, corrected_y = corrected_state[0, 0], corrected_state[1, 0]

