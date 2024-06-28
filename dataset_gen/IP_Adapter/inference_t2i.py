from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, DiffusionPipeline
import torch
from diffusers.utils import load_image
import random


pipeline = DiffusionPipeline.from_pretrained("/home/weights/sdxl_1_0", torch_dtype=torch.float32, use_safetensors=True).to("cuda")

for i in range(100):
    #seed = random.seed(i)
    seed = random.randint(1, 44455201144)
    print(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed) 
    images = pipeline(
        prompt='constuction site, tower crane, workers, truck, excavator, best quality',
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=50,
        generator=generator,
        guidance_scale=5,
    ).images[0]
    images.save("./generated_images/" + str(seed) + ".png")


# model_id = "/home/weights/sd1-5/runwayml/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")