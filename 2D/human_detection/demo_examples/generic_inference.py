import argparse
import os

import numpy as np
from PIL import Image

from trex import TRex2APIWrapper, visualize


def get_args():
    parser = argparse.ArgumentParser(description="Interactive Inference")
    parser.add_argument(
        "--token",
        type=str,
        help="The token for T-Rex2 API. We are now opening free API access to T-Rex2",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="The threshold for box score"
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="demo_vis/",
        help="The directory for visualization",
    )
    return parser.parse_args()

xywh_box = np.array([1938,851,46.27,56.95]
                    [1838,838,42.71,56.95]
                    [1669,781,46.27,62.29]
                    [2085,945,40.55,49.56]
                    [2165,806,23.28,30.04]
                    [2372,945,30.04,34.54]
                    [2040,564,53.29,71.05]
                    [2439,870,35.64,38.45]
                    [2300,1256,42.2,39.39]
                    [2434,1223,34.7,60.96]
                    [2371,1222,27.2,26.26]
                    [2411,727,37.14,55.11]
                    [2751,739,35.26,49.23]
                    [2566,1380,43.25,46.1]
                    [2535,1624,54.63,50.08]
                    [2652,1469,61.66,60.54]
                    [2810,1622,70.63,42.6]
                    [2948,1523,58.23,47.34]
                    [2982,1487,59.5,44.81]
                    [2588,1888,52.65,51.3]
                    [2185,1492,27.62,26.22]
                    [2370,2063,51.92,56.24]
                    [1192,219,36.56,41.91])

xyxy_box = np.ones_like(xywh_box)
xyxy_box[:, 0] = xywh_box[:, 0] - (xywh_box[:, 2]/2)
xyxy_box[:, 1] = xywh_box[:, 1] - (xywh_box[:, 3]/2)
xyxy_box[:, 2] = xywh_box[:, 0] + (xywh_box[:, 2]/2)
xyxy_box[:, 3] = xywh_box[:, 1] + (xywh_box[:, 3]/2)

xyxy_box.astype(int)
                    
if __name__ == "__main__":
    args = get_args()
    token = "338617ad80d8722e03e535c32709306b"
    trex2 = TRex2APIWrapper(token)
    target_image = "assets/trex2_api_examples/generic_target.jpg"
    prompts = [
        {
            "prompt_image": "assets/tower_crane/camera2_400_png.rf.db3bd8894c292752e884dbf3386e18d9.jpg",
            "rects": xyxy_box,
        }
    ]
    result = trex2.generic_inference(target_image, prompts)
    # filter out the boxes with low score
    scores = np.array(result["scores"])
    labels = np.array(result["labels"])
    boxes = np.array(result["boxes"])
    filter_mask = scores > args.box_threshold
    filtered_result = {
        "scores": scores[filter_mask],
        "labels": labels[filter_mask],
        "boxes": boxes[filter_mask],
    }
    # visualize the results
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    image = Image.open(target_image)
    image = visualize(image, filtered_result, draw_score=True)
    image.save(os.path.join(args.vis_dir, f"generic.jpg"))
    print(f"Visualized image saved to {args.vis_dir}/generic.jpg")


