from pathlib import Path
import numpy as np


model_path = Path('/home/Tower_crane_perception/2D/runs/train/exp/weights/last.pt')
video_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/hikrobot/")
video_with_human_path =  Path("/home/Tower_crane_perception/2D/human_detection/demo_vis")
lidar_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/livox/")
save_path = Path("/home/Tower_crane_perception/2D/runs/inference/2d_lidar_livox/")


USE_DEVICE = 2 # 0:Upper, 1:Lower, 2:Livox
## Upper camera and upper lidar

UpperDahua2UpperOuster = np.array([[ -0.00380224, -0.999927, -0.0114549, 0.0468632],
                                [-0.480837, 0.011872, -0.87673, 0.810458],
                                [0.876802, 0.00217442, -0.480847, 0.603057],
                                [0,                   0,               0,             1]])

UpperDahua_Intrinsic = np.array([[ 3727.035325,    0.000000000,   1905.471636],
                                 [0.000000000,    3707.139075,    1079.386277],
                                 [0.000000000,    0.000000000,    1.000000000]])
UpperDahua_dist = np.array([-0.4634329371, 0.3049578476, 0.01125169038, -0.003694136270, -0.2437815753])


## Lower camera and lower lidar

LowerDahua_Intrinsic = np.array([[3398.232603, 0, 1895.977318],
                      [0, 3409.324671, 1081.942343],
                      [0,0,1]])

LowerDahua_dist = np.array([-0.3873637993, 0.2923534322, -2.786784398e-05, -0.0001144799345, -0.2634945271])

LowerDahua2LowerOuster = np.array([[-0.00380224, -0.999927, -0.0114549, 0.0468632],
                [-0.480837, 0.011872, -0.87673, 0.810458],
                [0.876802, 0.00217442, -0.480847, 0.603057],
                [0,0,0,1]])

MVS_Intrinsic = np.array([[2601.379733,    0.000000000,    2679.106089],
                           [0.000000000,    2604.463961,    1851.265404],
                           [0.000000000,    0.000000000,    1.000000000]])

## MVS camera and Livox lidar

MVS_dist = np.array([-0.1281444408,    0.1072910023,    0.001436884538,    -0.0009784912760,    -0.05044686749])
MVS2Livox = np.array([[-0.00949657, 0.0142872, -0.999853, 0.00322139],
                      [-0.00300755, 0.999893,  0.0143163,    -0.121013],
                      [0.99995,0.00314306,-0.00945258,0.0338244],
                      [0,                 0,                 0,                 1]])
if USE_DEVICE == 0:
    Intrinsic = UpperDahua_Intrinsic
    c2L = UpperDahua2UpperOuster
    camera_dist = UpperDahua_dist
elif USE_DEVICE == 1:
    Intrinsic = LowerDahua_Intrinsic
    c2L = LowerDahua2LowerOuster
    camera_dist = LowerDahua_dist
elif USE_DEVICE == 2:
    Intrinsic = MVS_Intrinsic
    c2L = MVS2Livox
    camera_dist = MVS_dist
else:
    Intrinsic = None
    c2L = None
    camera_dist = None
    raise NotImplementedError