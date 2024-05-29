import os
import struct

def pcd_to_bin(pcd_folder, bin_folder):
    """
    Converts PCD files in the specified folder to BIN format.
    
    Args:
        pcd_folder (str): Path to the folder containing PCD files.
        bin_folder (str): Path to the folder where BIN files will be saved.
    """
    for filename in os.listdir(pcd_folder):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(pcd_folder, filename)
            bin_path = os.path.join(bin_folder, filename.replace(".pcd", ".bin"))

            # Read PCD file
            with open(pcd_path, "r") as pcd_file:
                lines = pcd_file.readlines()

            # Find DATA field
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("DATA"):
                    data_start = i + 1
                    break

            # Extract point cloud data
            point_cloud_data = []
            for i in range(data_start, len(lines)):
                values = lines[i].split()
                if len(values) == 3:
                    x, y, z = map(float, values)
                    point_cloud_data.append((x, y, z))

            # Write point cloud data to BIN file
            with open(bin_path, "wb") as bin_file:
                for point in point_cloud_data:
                    x, y, z = point
                    bin_file.write(struct.pack("fff", x, y, z))

            print(f"Converted {filename} to BIN format.")

# Example usage
# pcd_folder_path = "/home/haochen/HKCRC/3D_object_detection/data/site_data/test1/point_cloud_data_700_800"
# bin_folder_path = "/home/haochen/HKCRC/3D_object_detection/data/site_data/test1/point_cloud_data_700_800_bin"

pcd_folder_path = "/media/haochen/Drive1/HKCRC/site_data/test1/point_cloud_data_700_800_ROI"
bin_folder_path = "/media/haochen/Drive1/HKCRC/site_data/test1/point_cloud_data_700_800_ROI_bin_try"

pcd_to_bin(pcd_folder_path, bin_folder_path)
