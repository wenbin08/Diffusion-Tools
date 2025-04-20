# Diffusion-Tools

# 配置环境

## 克隆项目

```
git clone https://github.com/wenbin08/Diffusion-Tools.git
cd Diffusion-Tools
```



## 安装资源包

本机使用Pytorch 2.1.2   Python 3.10    CUDA 11.8

```
pip install -r requirements.txt
```



# 如何使用

以使用`sdxl`为例

```
python demo/sdxl.py
```



**因autodl下载huggingface模型自动下载到系统盘，建议先在数据盘下载模型，然后更改模型路径**



## 下载模型方式

```
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-base-1.0 --local-dir /root/autodl-tmp/SDXL
```

## 更改路径

```
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe = DiffusionPipeline.from_pretrained("/root/autodl-tmp/SDXL", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
```

