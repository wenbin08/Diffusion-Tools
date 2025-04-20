import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=40,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]

image.save("sd3_hello_world.png")
