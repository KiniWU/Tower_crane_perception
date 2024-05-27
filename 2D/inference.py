import torch

# Model 
#model = torch.hub.load("/home/2D/weights", "last.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path='last.pt', source='local')
# Images
img = "/home/2D/site_data_test3/test/images/camera1_250_png.rf.d07a9b287cff3a566148a274e34e051f.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.
