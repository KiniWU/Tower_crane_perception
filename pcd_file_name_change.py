import os
import shutil

def add_zero_padding_name(num):
    if num < 10:
        return "00000" + str(num) + ".bin"
    elif num >= 10 and num < 100:
        return "0000" + str(num) + ".bin"
    elif num >= 100 and num < 1000:
        return "000" + str(num) + ".bin"
    else:
        print("num is too large!!")
        return None
    
# get the file name list to nameList
path = "/media/itx4090/系统/code/dataset2/point_cloud_data_700_800_ROI_bin/"
save_path = "/media/itx4090/系统/code/dataset2/training/velodyne/"
if not os.path.exists(save_path):
    print("created")
    os.mkdir(save_path)
nameList = os.listdir(path) 
#loop through the name and rename
for fileName in nameList:
    if fileName.split(".")[-1] == "bin":
        rename=int(fileName.split(".")[0].split("_")[-1])-701
        print(path + fileName, rename)
        print(save_path)
        os.rename(path + fileName, save_path + add_zero_padding_name(rename))

