from roboflow import Roboflow
rf = Roboflow(api_key="1XSHUPUyuJghQQjMgukH")
project = rf.workspace("hkcrctowercrane").project("hkcrc_mic")
version = project.version(3)
dataset = version.download("yolov5")
