import argparse
import os

import numpy as np
from PIL import Image

from trex import TRex2APIWrapper, visualize
import cv2
from pathlib import Path

ORI_RESO = True 

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



              
if __name__ == "__main__":
    xywh_box = np.array([[1938,851,46.27,56.95],
                        [1838,838,42.71,56.95],
                        [1669,781,46.27,62.29],
                        [2085,945,40.55,49.56],
                        [2165,806,23.28,30.04],
                        [2372,945,30.04,34.54],
                        [2040,564,53.29,71.05],
                        [2439,870,35.64,38.45],
                        [2300,1256,42.2,39.39],
                        [2434,1223,34.7,60.96],
                        [2371,1222,27.2,26.26],
                        [2411,727,37.14,55.11],
                        [2751,739,35.26,49.23],
                        [2566,1380,43.25,46.1],
                        [2535,1624,54.63,50.08],
                        [2652,1469,61.66,60.54],
                        [2810,1622,70.63,42.6],
                        [2948,1523,58.23,47.34],
                        [2982,1487,59.5,44.81],
                        [2588,1888,52.65,51.3],
                        [2185,1492,27.62,26.22],
                        [2370,2063,51.92,56.24],
                        [1192,219,36.56,41.91]])

    # xywh_box = np.array([[205,920,80,85],
    #                     [780,15,45,90],
    #                     [1045,80,47.5,67.5],
    #                     [818,8,55,87.5],
    #                     [1085,528,52.5,67.5],
    #                     [830,98,55,82.5],
    #                     [648,690,67.5,90],
    #                     [398,995,75,102.5],
    #                     [673,895,72.visualize5,87.5],
    #                     [878,913,60,90],
    #                     [1170,1065,60,72.5],
    #                     [688,1343,87.5,145],
    #                     [483,195,50,55],
    #                     [949,517,56.51,90.42]])
    
    xyxy_box = np.ones_like(xywh_box)
    xyxy_box[:, 0] = xywh_box[:, 0]
    xyxy_box[:, 1] = xywh_box[:, 1]
    xyxy_box[:, 2] = xywh_box[:, 0] + xywh_box[:, 2]
    xyxy_box[:, 3] = xywh_box[:, 1] + xywh_box[:, 3]

    xyxy_box = xyxy_box.astype(int)
    # print(xyxy_box)

    # #test_im = cv2.imread("/home/Tower_crane_perception/2D/human_detection/assets/trex2_api_examples/generic_prompt1.jpg")
    # test_im = cv2.imread("/home/Tower_crane_perception/2D/human_detection/assets/tower_crane/camera1_36-1-_png.rf.dd69389d01a93eefa2b2a7e0300ff174.jpg")
    # #test_im = cv2.rectangle(test_im, (692, 338), (725, 459), color=(0, 255, 0), thickness=2)
    # test_im = cv2.rectangle(test_im, (xyxy_box[1, 0], xyxy_box[1, 1]), (xyxy_box[1, 2], xyxy_box[1, 3]), color=(0, 255, 0), thickness=2)
    # cv2.imwrite("test_im.png", test_im)

    args = get_args()
    token = "338617ad80d8722e03e535c32709306b"
    trex2 = TRex2APIWrapper(token)

    video_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/hikrobot/")
    lidar_path = Path("/home/tower_crane_data/site_data/test4/sync_camera_lidar/livox/")
    image_list = sorted(video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
    
    save_path = Path("/home/Tower_crane_perception/2D/runs/inference/2d_lidar_livox_human/")
    save_path.mkdir(exist_ok=True, parents=True)

    if ORI_RESO:
        size = (5472, 3648)
    else:
        size = (1344, 768)
    #out = cv2.VideoWriter(str(save_path / "video.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 10, size, True)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True) 
    prompts = [
            {
                "prompt_image":  "assets/tower_crane/camera2_400_png.rf.db3bd8894c292752e884dbf3386e18d9.jpg", #"assets/tower_crane/camera1_36-1-_png.rf.dd69389d01a93eefa2b2a7e0300ff174.jpg", #
                "rects": xyxy_box,
            }
        ]

    x_ratio = 5472 / 1344
    y_ratio = 3648 / 768
    
    for n, i_p  in enumerate(image_list[9:50]):

        target_image = str(i_p)
        # target_image = "assets/tower_crane/camera1_36-1-_png.rf.dd69389d01a93eefa2b2a7e0300ff174.jpg"
        
        print("begin inference")
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
        # if not os.path.exists(args.vis_dir):
        #     os.makedirs(args.vis_dir)
        image = Image.open(target_image)
        image = visualize(image, filtered_result, draw_score=True, random_color=False, draw_width=12)
        #image_to_save = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image.save(os.path.join(args.vis_dir, f"generic" + str(n+9) + ".jpg"))
        # print(f"Visualized image saved to {args.vis_dir}/generic.jpg")

        # out.write(image_to_save)

# cv2.destroyAllWindows()
# out.release()
