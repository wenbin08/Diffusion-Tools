from diffusers import DiffusionPipeline
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")

tokenizer = pipe.tokenizer

prompt = "An astronaut riding a green horse"

# 仅仅为了展示真实的prompt，实际不需要tokenizer, decode
prompts = tokenizer(
    prompt,
    max_length=77,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).input_ids

step_texts_tokenized = tokenizer.batch_decode(prompts, skip_special_tokens=True)
print(f"decode:{step_texts_tokenized}")

output_images = pipe(
    step_texts_tokenized,
    height=512,
    width=512,
    num_inference_steps=50,
).images


output_images[0].save("output.png")
print(f"Saving image to output.png")
