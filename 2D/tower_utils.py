import cv2
import numpy as np

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

MVS_dist = np.array([-0.1281444408,    0.1072910023,    0.001436884538,    -0.0009784912760,    -0.05044686749])
MVS2Livox = np.array([[-0.00949657, 0.0142872, -0.999853, 0.00322139],
                      [-0.00300755, 0.999893,  0.0143163,    -0.121013],
                      [0.99995,0.00314306,-0.00945258,0.0338244],
                      [0,                 0,                 0,                 1]])

Intrinsic = MVS_Intrinsic
c2L = MVS2Livox
camera_dist = MVS_dist

def pixel2Camera(pixel_pt=[], distance = 1.0):
    """
    pixel_pt: x,y,1
    distance: [m]
    """
    n_pt = cv2.undistortPoints(np.array([[pixel_pt[:-1]]]), Intrinsic, camera_dist, P=Intrinsic)
    #print(n_pt)
    n_pt = np.dot(np.linalg.inv(Intrinsic), pixel_pt)

    ratio = distance / np.sqrt(n_pt[0]**2 + n_pt[1]**2 + n_pt[2]**2)

    return n_pt*ratio

def camera2Lidar(camera_pt=[]):
    c_pt = np.ones((4, 1), np.float32)
    c_pt[:3, 0] = camera_pt
    return np.dot(c2L, c_pt)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1=[], v2=[]):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_closest_cluster(vecs=[[]], vec1=[], n=2):
    angles = []
    for i in range(len(vecs)):
        angle = angle_between_vectors(vecs[i], vec1)
        angles.append(angle)
    print(angles)
    return np.argmin(angles) #, np.argpartition(angles,n-1)[:n]

def get_3d_box_from_points(pts=[]):
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])

    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])


    return (x_min, x_max, y_min, y_max, z_min, z_max)