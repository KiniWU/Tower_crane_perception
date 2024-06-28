# from roboflow import Roboflow
# rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
# project = rf.workspace("hkcrctowercrane").project("hkcrc-mic-mvs")
# version = project.version(1)
# dataset = version.download("yolov5")


# hkcrc-people-mvs
# from roboflow import Roboflow
# rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
# project = rf.workspace("hkcrctowercrane").project("hkcrc-people-detection")
# version = project.version(1)
# dataset = version.download("yolov5")

#333 dataset v1 download 
# !pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
# project = rf.workspace("hkcrctowercrane").project("333-mszug")
# version = project.version(1)
# dataset = version.download("yolov5")

#333 dataset v3 (2024-06-24 8:58am) download 
# !pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
# project = rf.workspace("hkcrctowercrane").project("333-mszug")
# version = project.version(3)
# dataset = version.download("yolov5")

#333 dataset v4 (2024-06-26 7:07am) download 
# !pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
project = rf.workspace("hkcrctowercrane").project("333-mszug")
version = project.version(4)
dataset = version.download("yolov5")
