#!/bin/bash

# 指定Python脚本的路径
SCRIPT_PATH="./infer.py"

# 指定模型路径（从Hugging Face获取）
MODEL_PATH="/home/humw/Pretrains/stable-diffusion-v1-4"

# 指定输出目录
OUTPUT_DIR="./generated_images"

# 调用Python脚本并传递参数
python $SCRIPT_PATH --model_path $MODEL_PATH --output_dir $OUTPUT_DIR