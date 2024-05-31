import numpy as np
import open3d as o3d
import progressbar
import matplotlib.pyplot as plt

# Load the PCD file
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# Downsample the point cloud
def downsample_point_cloud(pcd, voxel_size=0.05):
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return down_pcd

# Cluster the point cloud
def cluster_point_cloud(pcd, eps=1, min_points=100):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    return labels

# Visualize the clusters
def visualize_clusters(pcd, labels):
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Set black color for noise points
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def get_cluster_info(pcd_file, dbscan_labels):
    # read .pcd file 
    # pcd = o3d.io.read_point_cloud(pcd_file)
    #points = np.asarray(pcd.points)

    points = np.asarray(pcd_file.points)
    
    # combine each point and its cluster type
    points_with_labels = np.hstack((points, dbscan_labels.reshape(-1, 1)))

    # calculate cluster center
    cluster_centers = []
    unique_labels = np.unique(dbscan_labels)
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            cluster_points = points[dbscan_labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
    
    # convert cluster center to array
    cluster_centers = np.array(cluster_centers)
    
    return cluster_centers, points_with_labels


# Main function
def main():
    #file_path = '/home/haochen/HKCRC/3D_object_detection/Tower_crane_perception/data_process/ouster_1.pcd'  # Replace with your file path
    file_path = '/home/haochen/HKCRC/3D_object_detection/Tower_crane_perception/data_process/ouster1_300.pcd'  # Replace with your file path
    pcd = load_point_cloud(file_path)
    
    # Initialize the progress bar
    bar = progressbar.ProgressBar(maxval=pcd.points.__len__(), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # Downsample the point cloud
    down_pcd = downsample_point_cloud(pcd)
    
    # Cluster the point cloud
    labels = cluster_point_cloud(down_pcd)
    # with open("my_list.txt", "w") as file:
    #     for item in labels:
    #         file.write(f"{labels}\n")
    # print(labels)
    
    # Update the progress bar
    for i in range(down_pcd.points.__len__()):
        bar.update(i+1)
    
    bar.finish()
    
    # Visualize the original point cloud
    print('Displaying the original point cloud...')
    o3d.visualization.draw_geometries([down_pcd])
    
    # Visualize the clustered results
    print('Displaying the clustered results...')
    visualize_clusters(down_pcd, labels)

    cluster_centers, points_with_labels = get_cluster_info(down_pcd, labels)
    print(cluster_centers)
    print(points_with_labels)

    # with open("points_with_labels.txt", "w") as file:
    #     for label in points_with_labels:
    #         file.write(f"{label}\n")

if __name__ == '__main__':
    main()
