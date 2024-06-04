import numpy as np


def tracking_diff(previous_pts, back = 3):
    previous_pts = np.array(previous_pts)
    diff_x, diff_y = np.mean(previous_pts[-1*back:-1, :] - previous_pts[-1*(back+1):-2, :], dim=1)

    return (previous_pts[-1, 0] + diff_x, previous_pts[-1, 1] + diff_y)