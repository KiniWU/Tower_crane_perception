from diffusers import AutoPipelineForText2Image, \
    StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline, \
        StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
import json
from diffusers.utils import load_image
from PIL import Image, ImageFilter
import random
from pathlib import Path
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("/home/weights/sdxl_1_0", # "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/", 
                                                      #use_safetensors=True,
                                                      torch_dtype=torch.float16).to("cuda")
save_path = Path("/home/tower_crane_data/gen_dataset/generated_mic/")
save_path.mkdir(parents=True, exist_ok=True)
# pipeline.load_ip_adapter("/home/weights/trained_models/ip_adapter_sdxl_mic_only", subfolder="checkpoint-80000/", weight_name="ip_adapter.bin")
# #print(pipeline)
# pipeline.set_ip_adapter_scale(0.1)

init_image_path = Path("/home/tower_crane_data/gen_dataset/333-v3/train/images")
init_image_list = list(init_image_path.rglob("*.jpg"))

anno_data = json.load(open("/home/tower_crane_data/gen_dataset/333-v3/mic_only_dataset.json"))

ip_adapter_image_path = Path("/home/tower_crane_data/gen_dataset/333-v3/cropped")
ip_adapter_image_list = list(ip_adapter_image_path.rglob("*.jpg"))
#image = load_image("/home/tower_crane_data/gen_dataset/333-v2/train/images/hik_11_png.rf.8f08b79aea46d27122c064d8710083b0.jpg")

for i in range(100):
    # init_im_path = random.choice(init_image_list)
    # init_image = load_image(str(init_im_path))
    # init_image = init_image.resize((1368, 912))

    anno_data_selected = random.choice(anno_data)
    init_image = load_image(str(anno_data_selected["image_file"]))
    #init_image.save("init_image.png")
    bbox = anno_data_selected["bbox"]
    x_min, y_min, x_max, y_max = bbox
    mask = np.ones_like(init_image)*255
    mask[y_min:y_max, x_min:x_max, :] = 0
    mask = mask[:, :, 0]
    mask = Image.fromarray(mask)
    #mask.save("mask.png")
    print(mask.size, init_image.size)
    init_image = init_image.resize((1368, 912))
    mask = mask.resize((1368, 912))
    #init_image = init_image.filter(ImageFilter.GaussianBlur(5))

    ip_im_path = random.choice(ip_adapter_image_list)
    image = load_image(str(ip_im_path))
    seed = random.randint(1, 44455201144)
    print(ip_im_path, seed)
    generator = torch.Generator(device="cuda").manual_seed(seed) 
    images = pipeline(
        image=init_image,
        mask_image=mask,
        prompt='A Modular Integrated Construction lift on the construction site, beside the sea, a lot of workers, high view',
        prompt_2='beside the sea, a lot of workers, high view',
        #ip_adapter_image=image,
        negative_prompt="",
        num_inference_steps=30,
        generator=generator,
        guidance_scale=7,
    ).images[0]
    images.save(str(save_path / ("mic" + str(i) + '.png')))


# model_id = "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")