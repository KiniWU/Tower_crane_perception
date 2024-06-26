from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image, make_image_grid

model_id = "/home/Tower_crane_perception/dataset_gen/Diffusers/examples/dreambooth/dreambooth_mic"
pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks mic beside the sea"
#init_image = load_image("/home/tower_crane_data/gen_dataset/dreambooth_dataset/hik_5_png.rf.3100dc395474b06f9d0c669bfe09e2ed.jpg")
init_image = load_image("/home/tower_crane_data/gen_dataset/333-v3/train/images/hik_6_png.rf.e761e35515aa6b0d070a29a3880e37c0.jpg")
init_image = init_image.resize((1368, 912))
image = pipe(prompt, image=init_image, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("mic_sea.png")