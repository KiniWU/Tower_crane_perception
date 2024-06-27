import cv2
import numpy as np

def KF_tracking(measured_state):
    # init kalman filter
    kalman_filter = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements 
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix  = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman_filter.processNoiseCov   = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4

    #measured_state = np.array([[measured_x], [measured_y]], np.float32)

    # prediction
    predicted_state = kalman_filter.predict()

    # correction
    corrected_state = kalman_filter.correct(measured_state)

    corrected_x, corrected_y = corrected_state[0, 0], corrected_state[1, 0]

    return corrected_state


def sliding_window_filter(positions, window_size=5):

    window = np.ones(window_size) / window_size

    filtered_positions = []

    for dimension in zip(*positions):
        filtered_dimension = np.convolve(dimension, window, mode='same')
        filtered_positions.append(filtered_dimension)

    return np.array(filtered_positions).T
