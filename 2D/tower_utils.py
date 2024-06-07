import cv2
import numpy as np
from config import Intrinsic, c2L, camera_dist


def pixel2Camera(pixel_pt=[], distance = 1.0):
    """
    pixel_pt: x,y,1
    distance: [m]
    """
    n_pt = cv2.undistortPoints(np.array([[pixel_pt[:-1]]]), Intrinsic, camera_dist, P=Intrinsic)
    print(n_pt)
    new_pt = np.array([n_pt[0, 0, 0], n_pt[0, 0, 1], 1])
    n_pt = np.dot(np.linalg.inv(Intrinsic), new_pt)

    ratio = distance / np.sqrt(n_pt[0]**2 + n_pt[1]**2 + n_pt[2]**2)

    return n_pt*ratio

def camera2Pixel(camera_pt=[]):
    """
    camera_pt: x,y,z,1
    """
    c_pt = np.ones((3, camera_pt.shape[-1]), np.float32)
    c_pt = camera_pt[:3, :] / camera_pt[2, :]
    
    return np.dot(Intrinsic, c_pt)

def lidar2Camera(lidar_pt=[]):
    return np.dot(np.linalg.inv(c2L), lidar_pt)

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

def find_closest_cluster_angle(vecs=[[]], vec1=[], n=2):
    angles = []
    for i in range(len(vecs)):
        angle = angle_between_vectors(vecs[i], vec1)
        angles.append(angle)
    print(angles)
    return np.argmin(angles) #, np.argpartition(angles,n-1)[:n]

def find_closest_cluster_eucli(vecs=[[]], vec1=[]):
    """
    vecs = [[1, a],
            [2, b]
            [3, c],
            [.., ..]]
    vec1 = [[1],
            [2]
            [3],
            [..]]
    """
    assert vecs.shape[0] == vec1.shape[0]
    diff_sqrt = (vecs - vec1) * (vecs - vec1) 
    # print("test math", vecs, vec1, (vecs - vec1))
    # print("test math", diff_sqrt)
    
    dis = np.sum(diff_sqrt, axis=0)
    # print("test math", dis)
    return np.argmin(dis) #, np.argpartition(angles,n-1)[:n]

def get_3d_box_from_points(pts=[]):
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])

    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])


    return (x_min, x_max, y_min, y_max, z_min, z_max)

def draw_3d_box(img, threed_box, line_thickness = 10, line_color_c = (0,0,255)):
    (x_min, x_max, y_min, y_max, z_min, z_max) = threed_box
    point0 = np.array((x_min, y_min, z_min, 1), np.float32).reshape(4, 1)
    point0_camera = lidar2Camera(point0)
    point0_pixel = camera2Pixel(point0_camera).astype(int)
    point1 = np.array((x_min, y_min, z_max, 1), np.float32).reshape(4, 1)
    point1_camera = lidar2Camera(point1)
    point1_pixel = camera2Pixel(point1_camera).astype(int)

    point2 = np.array((x_min, y_max, z_min, 1), np.float32).reshape(4, 1)
    point2_camera = lidar2Camera(point2)
    point2_pixel = camera2Pixel(point2_camera).astype(int)
    point3 = np.array((x_min, y_max, z_max, 1), np.float32).reshape(4, 1)
    point3_camera = lidar2Camera(point3)
    point3_pixel = camera2Pixel(point3_camera).astype(int)

    point4 = np.array((x_max, y_min, z_min, 1), np.float32).reshape(4, 1)
    point4_camera = lidar2Camera(point4)
    point4_pixel = camera2Pixel(point4_camera).astype(int)
    point5 = np.array((x_max, y_min, z_max, 1), np.float32).reshape(4, 1)
    point5_camera = lidar2Camera(point5)
    point5_pixel = camera2Pixel(point5_camera).astype(int)

    point6 = np.array((x_max, y_max, z_min, 1), np.float32).reshape(4, 1)
    point6_camera = lidar2Camera(point6)
    point6_pixel = camera2Pixel(point6_camera).astype(int)
    point7 = np.array((x_max, y_max, z_max, 1), np.float32).reshape(4, 1)
    point7_camera = lidar2Camera(point7)
    point7_pixel = camera2Pixel(point7_camera).astype(int)

    line_thickness = 10
    line_color_c = (0,255,0)
    print((point0_pixel[0][0], point0_pixel[1][0], point1_pixel[0][0], point1_pixel[1][0]))
    img = cv2.line(img, (point0_pixel[0][0], point0_pixel[1][0]), (point1_pixel[0][0], point1_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point1_pixel[0][0], point1_pixel[1][0]), (point2_pixel[0][0], point2_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point2_pixel[0][0], point2_pixel[1][0]), (point3_pixel[0][0], point3_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point3_pixel[0][0], point3_pixel[1][0]), (point4_pixel[0][0], point4_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point4_pixel[0][0], point4_pixel[1][0]), (point5_pixel[0][0], point5_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point5_pixel[0][0], point5_pixel[1][0]), (point6_pixel[0][0], point6_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point6_pixel[0][0], point6_pixel[1][0]), (point7_pixel[0][0], point7_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point7_pixel[0][0], point7_pixel[1][0]), (point0_pixel[0][0], point0_pixel[1][0]), line_color_c, line_thickness)

    return img