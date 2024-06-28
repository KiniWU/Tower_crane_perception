from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
from diffusers.utils import load_image
import random


pipeline = StableDiffusionPipeline.from_single_file("/home/weights/sd1-5/juggernaut.safetensors", # "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/", 
                                                      use_safetensors=True,
                                                      torch_dtype=torch.float32).to("cuda")
# pipeline = StableDiffusionPipeline.from_pretrained("/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/", 
#                                                       use_safetensors=True,
#                                                       torch_dtype=torch.float32).to("cuda")
pipeline.load_ip_adapter("/home/weights/trained_models/ip_adapter_mic_only_v3/", subfolder="checkpoint-12000/", weight_name="ip_adapter.bin")
#print(pipeline)
pipeline.set_ip_adapter_scale(1.0)

image = load_image("/home/tower_crane_data/gen_dataset/333-v3/cropped/hik_26_png.rf.e24aadfe1521027cf53bde3d7bbf512d.jpg")
seed = random.randint(1, 44455201144)
print(seed)
generator = torch.Generator(device="cuda").manual_seed(seed) 
images = pipeline(
    prompt='A Modular Integrated Construction on the construction site beside the sea',
    ip_adapter_image=image,
    negative_prompt="",
    num_inference_steps=80,
    generator=generator,
    guidance_scale=5,
).images[0]
images.save("tower_crane.png")


# model_id = "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")