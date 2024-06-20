from pathlib import Path
import json
from config import ip_adapter_dataset_path, mic_dataset_path
import cv2


images = (mic_dataset_path / "train/images").rglob("*.jpg")
bbox_path = mic_dataset_path / "train/labels"
cropped_images_path = mic_dataset_path / "cropped"
cropped_images_path.mkdir(exist_ok=True, parents=True)
dataset = []

for i_p in images:
    print(i_p)
    image = cv2.imread(str(i_p))
    height, width, _ = image.shape
    bbox_path = mic_dataset_path / "train/labels"
    with open(bbox_path/(i_p.name[:-3] + "txt"), "r") as f:
        lines = f.readlines()
        for l in lines:
            l_text = l.split(" ")
            if l_text[0] == "1":
                bbox = [float(x) for x in l_text[1:]]
                print(bbox)
                x_min = int((bbox[0] - bbox[2]/2) * width)
                x_max = int((bbox[0] + bbox[2]/2) * width)

                y_min = int((bbox[1] - bbox[3]/2) * height)
                y_max = int((bbox[1] + bbox[3]/2) * height)
                im = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 10)
                cv2.imwrite(cropped_images_path / i_p.name, im)

                # print(x_min, y_min, x_max, y_max)

                one_anno = {"image_file": str(i_p), "text": "A MIC on the constuction site", "bbox": [x_min, y_min, x_max, y_max]}
                dataset.append(one_anno)


# data = [{"image_file": "1.png", "text": "A MIC on the constuction site"},
#         {"image_file": "1.png", "text": "A MIC on the constuction site"},
#         {"image_file": "1.png", "text": "A MIC on the constuction site"}]

gen_dataset = ip_adapter_dataset_path

gen_dataset.write_text(json.dumps(dataset, indent=4))