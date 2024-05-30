# https://ai-robotic.de/exploring-clustering-and-visualization-of-3d-point-cloud-datausing-python/
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def get_cluster_info(pcd_file, dbscan_labels):
    # 读取点云数据
    # pcd = o3d.io.read_point_cloud(pcd_file)
    #points = np.asarray(pcd.points)
    
    # 初始化簇中心点坐标矩阵和点坐标及簇类型矩阵
    cluster_centers = []
    points_with_labels = np.hstack((pcd_file, dbscan_labels.reshape(-1, 1)))
    # points_with_labels = []

    # 计算每个簇的中心点坐标
    unique_labels = np.unique(dbscan_labels)
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            cluster_points = points[dbscan_labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
    
    # 将簇中心点坐标转换为numpy数组
    cluster_centers = np.array(cluster_centers)
    
    return cluster_centers, points_with_labels

#Generating Synthetic Clusters
np.random.seed(1)
num_points = 30
cluster_params = [
 {"mean": np.array([0, 0, 0]),    "cov": np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])},
 {"mean": np.array([4, 4, 4]),    "cov": np.array([[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]])},
 {"mean": np.array([-3, -4, -5]), "cov": np.array([[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]])}
]

clusters = []
for param in cluster_params:
 cluster = np.random.multivariate_normal(param["mean"], param["cov"], num_points // 3)
#  print(cluster)
 clusters.append(cluster)
points = np.vstack(clusters)

# pcd = o3d.io.read_point_cloud("/home/haochen/HKCRC/3D_object_detection/Tower_crane_perception/data_process/ouster_1.pcd")
# points = np.asarray(pcd.points)  
# print("output array from input list : ", points)  

#Visualizing the Synthetic Point Cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Synthetic Point Cloud')
plt.show()

#Clustering with DBSCAN
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
eps        = 1.2 # Distance threshold for points in a cluster
min_points = 3 # Minimum number of points per cluster
# dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

print("Cluster labels (with -1 indicating noise): ")
print(f"Labels: {dbscan_labels}")

# Visualizing the Clustering Results
colors = plt.get_cmap("tab10")(dbscan_labels)
colors[dbscan_labels == -1] = [0.5, 0.5, 0.5, 1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

cluster_centers, points_with_labels = get_cluster_info(points, dbscan_labels)
print(cluster_centers)
print(points_with_labels)


# Write labels to a file
with open("points_with_labels.txt", "w") as file:
    for label in points_with_labels:
        file.write(f"{label}\n")