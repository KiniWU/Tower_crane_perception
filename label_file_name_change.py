import os


def add_zero_padding_name(num):
    if num < 10:
        return "00000" + str(num) + ".txt"
    elif num >= 10 and num < 100:
        return "0000" + str(num) + ".txt"
    elif num >= 100 and num < 1000:
        return "000" + str(num) + ".txt"
    else:
        print("num is too large!!")
        return None
    
# get the file name list to nameList
path = "/media/itx4090/系统/code/dataset2/training/label_2/"
save_path = "/media/itx4090/系统/code/dataset2/training/label_2_1/"
if not os.path.exists(save_path):
    print("created")
    os.mkdir(save_path)
nameList = os.listdir(path) 
#loop through the name and rename
for fileName in nameList:
    if fileName.split(".")[-1] == "txt":
        rename=int(fileName.split(".")[0])
        print(path + fileName, rename)
        print(save_path)
        os.rename(path + fileName, save_path + add_zero_padding_name(rename))

