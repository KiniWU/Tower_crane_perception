from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained("/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/", torch_dtype=torch.float32).to("cuda")
pipeline.load_ip_adapter("/home/weights/h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
#print(pipeline)

image = load_image("/home/tower_crane_data/dataset_mic/v4/train/images/camera1_27_png.rf.bcb87b2f5a35f4a2e117f005ec701992.jpg")
generator = torch.Generator(device="cpu").manual_seed(5654)
images = pipeline(
    prompt='best quality, high quality, construction site',
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50,
    generator=generator,
    guidance_scale=3,
).images[0]
images.save("tower_crane.png")


# model_id = "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")