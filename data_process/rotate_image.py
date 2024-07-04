import cv2
import os
from pathlib import Path

def rotate_image(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_files = len(os.listdir(input_dir))
    processed_files = 0

    for img_name in os.listdir(input_dir):
        print(img_name)
        
        img_path = os.path.join(input_dir, img_name)
        # read img
        img = cv2.imread(img_path)

        # rotate 180 deg
        rotated_180 = cv2.rotate(img, cv2.ROTATE_180)

        # save rotated image
        cv2.imwrite(os.path.join(output_dir, img_name), rotated_180)

        # update process
        processed_files += 1
        #print(f'process：{processed_files}/{total_files}')
        

if __name__=="__main__":
    input_dir  = '/home/tower_crane_data/crcust/crcust_top/mvs_avia/2024-06-28-11-15-24/pic'
    output_dir = '/home/tower_crane_data/crcust/crcust_top/mvs_avia/2024-06-28-11-15-24/pic_rotated'
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    rotate_image(input_dir,output_dir)

