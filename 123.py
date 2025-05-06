import torch
import os
import argparse
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline
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
    
    try:
        # 加载模型
        if args.model == "black-forest-labs/FLUX.1-dev":
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
        elif args.model == "stabilityai/stable-diffusion-3.5-large":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                args.model, 
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                args.model, 
                torch_dtype=torch.float16,
            ).to("cuda")
            print("haha")

        # 读取输入文件
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 生成图片
        for cnt, line in enumerate(lines):
            try:
                prompt = line.strip()
                print(prompt)
                if not prompt:
                    continue
                
                print(f"正在处理第 {cnt+1} 行: {prompt}")
                
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                )
                
                if hasattr(output, 'images') and output.images:
                    image = output.images[0]
                    output_path = os.path.join(args.output, f"img_{cnt}.png")
                    image.save(output_path)
                    print(f"已生成图片: {output_path}")
                else:
                    print(f"生成第 {cnt+1} 张图片失败: 无效的输出")
                    
            except Exception as e:
                print(f"处理第 {cnt+1} 行时出错: {str(e)}")
                continue
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return

if __name__ == "__main__":
    main()
