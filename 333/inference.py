import torch
import cv2

# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path='/home/Tower_crane_perception/2D/runs/train/exp2/weights/last.pt', source='local')
# Images
img_path = "/home/2D/site_data_test3/train/images/camera1_400_png.rf.729b700aca6f78e53df9d559e1f0438f.jpg"  # or file, Path, PIL, OpenCV, numpy, list

img = cv2.imread(img_path)
img = cv2.resize(img, (1344, 768), cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model.iou = 0.2
model.conf = 0.2
# Inference
results = model(img, size=1344)
print(results)
# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.
