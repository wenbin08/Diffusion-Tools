# Diffusion-Tools

# 配置环境

## 克隆项目

```
git clone https://github.com/wenbin08/Diffusion-Tools.git
cd Diffusion-Tools
```



## 安装资源包

```
pip install -r requirements.txt
```



# 如何使用

**因autodl下载huggingface模型自动下载到系统盘，建议先在数据盘下载模型，然后更改模型路径**

```
python run.py --model stabilityai/stable-diffusion-xl-base-1.0 --input test.txt --output ./outputs --height 512 --width 512 --steps 40
```

`--model`：模型路径 or Huggingface 模型ID

`--input`：输入文本文件路径

`--output`：输出图片目录

`--height --width`：输出图片高度/宽度

`--steps`：推理步数



## 下载模型方式

```
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-base-1.0 --local-dir /root/autodl-tmp/SDXL
```

**注意：本地模型文件夹命名，否则脚本会有问题**

`SDXL or SDxl` `SD35 or SD3.5` `FLUX or flux`
