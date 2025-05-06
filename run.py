import torch
import os
import argparse
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from diffusers import FluxPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion text to image generator')
    parser.add_argument('--model', type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help='模型路径或Hugging Face模型ID')
    parser.add_argument('--input', type=str, required=True,
                        help='输入文本文件路径')
    parser.add_argument('--output', type=str, default="./outputs",
                        help='输出图片保存目录')
    parser.add_argument('--height', type=int, default=512,
                        help='输出图片高度')
    parser.add_argument('--width', type=int, default=512,
                        help='输出图片宽度')
    parser.add_argument('--steps', type=int, default=40,
                        help='推理步数')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载模型
    if args.model == "black-forest-labs/FLUX.1-dev" or "FLUX" in args.model or "flux" in args.model:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    if args.model == "stabilityai/stable-diffusion-3.5-large" or "35" in args.model or "3.5" in args.model:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16
        ).to("cuda")
    if args.model == "stabilityai/stable-diffusion-xl-base-1.0" or "XL" in args.model or "xl" in args.model:
        pipe = DiffusionPipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.float16
        ).to("cuda")

    

    # 读取输入文件
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 生成图片
    for cnt, line in enumerate(lines):
        image = pipe(
            prompt=line.strip(),
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width
        ).images[0]

        # 保存图片
        output_path = os.path.join(args.output, f"img_{cnt}.png")
        image.save(output_path)
        print(f"已生成图片: {output_path}")

if __name__ == "__main__":
    main()
