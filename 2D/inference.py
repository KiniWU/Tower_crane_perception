import torch

# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path='/home/Tower_crane_perception/2D/runs/train/exp6/weights/best.pt', source='local')
# Images
img = "/home/2D/site_data_test3/train/images/camera1_300_png.rf.fff62f6ad0615431f9ffe2edfab52665.jpg"  # or file, Path, PIL, OpenCV, numpy, list

model.iou = 0.01
model.conf = 0.01
# Inference
results = model(img, size=1280)
print(results)
# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.
