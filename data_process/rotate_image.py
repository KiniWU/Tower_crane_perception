import cv2
import os

input_dir = '/home/haochen/HKCRC/tower_crane_data/site_data/test4/sync_camera_lidar/camera'
output_dir = '/home/haochen/HKCRC/tower_crane_data/site_data/test4/sync_camera_lidar/rotated_camera'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_files = len(os.listdir(input_dir))
processed_files = 0

for img_name in os.listdir(input_dir):
    
    img_path = os.path.join(input_dir, img_name)
    # read img
    img = cv2.imread(img_path)

    # rotate 180 deg
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)

    # save rotated image
    cv2.imwrite(os.path.join(output_dir, img_name), rotated_180)

    # update process
    processed_files += 1
    print(f'process：{processed_files}/{total_files}')

