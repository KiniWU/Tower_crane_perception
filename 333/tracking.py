import numpy as np

def KF_init(initial_state):
    # init kalman filter
    Delta_t = 1
    kalman_filter = cv2.KalmanFilter(4, 2)  # 3 states, 2 measurements 
    kalman_filter.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix    = np.array([[1, 0, Delta_t, 0], [0, 1, 0, Delta_t], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman_filter.processNoiseCov     = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1e-4
    kalman_filter.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
    kalman_filter.statePre            = initial_state
    kalman_filter.statePost           = initial_state
    return kalman_filter

def KF_tracking(kalman_filter,measured_state):
    # prediction
    predicted_state = kalman_filter.predict()

    # correction
    corrected_state = kalman_filter.correct(measured_state)

    return kalman_filter,corrected_state

def tracking_diff(previous_pts, back = 3):
    previous_pts = np.array(previous_pts)
    diff_x, diff_y = np.mean(previous_pts[-1*back:-1, :] - previous_pts[-1*(back+1):-2, :], dim=1)

    return (previous_pts[-1, 0] + diff_x, previous_pts[-1, 1] + diff_y)