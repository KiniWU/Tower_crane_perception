from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline
import torch
from diffusers.utils import load_image
import random
from pathlib import Path

pipeline = StableDiffusionXLPipeline.from_pretrained("/home/weights/sdxl_1_0", # "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/", 
                                                      #use_safetensors=True,
                                                      torch_dtype=torch.float16).to("cuda")
save_path = Path("/home/tower_crane_data/gen_dataset/generated_mic2/")
save_path.mkdir(parents=True, exist_ok=True)
pipeline.load_ip_adapter("/home/weights/trained_models/ip_adapter_sdxl_mic_only", subfolder="checkpoint-20000/", weight_name="ip_adapter.bin")
#print(pipeline)
pipeline.set_ip_adapter_scale(1.0)

ip_adapter_image_path = Path("/home/tower_crane_data/gen_dataset/333-v3/cropped")
ip_adapter_image_list = list(ip_adapter_image_path.rglob("*.jpg"))
#image = load_image("/home/tower_crane_data/gen_dataset/333-v2/train/images/hik_11_png.rf.8f08b79aea46d27122c064d8710083b0.jpg")

for i in range(100):
    ip_im_path = random.choice(ip_adapter_image_list)
    image = load_image(str(ip_im_path))
    seed = random.randint(1, 44455201144)
    print(ip_im_path, seed)
    generator = torch.Generator(device="cuda").manual_seed(seed) 
    images = pipeline(
        prompt='A Modular Integrated Construction lift on the construction site',
        ip_adapter_image=image,
        negative_prompt="",
        num_inference_steps=80,
        generator=generator,
        guidance_scale=3,
    ).images[0]
    images.save(str(save_path / ("mic" + str(i) + '.png')))


# model_id = "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")