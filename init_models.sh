#!/bin/bash
set -e

echo "Ensuring model directories exist..."
mkdir -p /app/models/vae
mkdir -p /app/models/clip
mkdir -p /app/models/clip_vision
mkdir -p /app/models/echo_mimic
mkdir -p /app/custom_nodes/comfyui-frame-interpolation/ckpts/rife/

echo "Downloading required models (this will be skipped if they already exist)..."

huggingface-cli download Kijai/Wan2.1-VAE-safetensors Wan2_1_VAE_bf16.safetensors --local-dir /app/models/vae
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_pretrained umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir /app/models/clip
huggingface-cli download comfyanonymous/clip_vision_g clip_vision_h.safetensors --local-dir /app/models/clip_vision

huggingface-cli download BadToBest/EchoMimicV3 --include "echomimicv3-flash-pro/*" --local-dir /app/models/echo_mimic
huggingface-cli download BadToBest/EchoMimicV3 --include "chinese-wav2vec2-base/*" --local-dir /app/models/echo_mimic
huggingface-cli download BadToBest/EchoMimicV3 --include "transformer/*" --local-dir /app/models/echo_mimic

if [ ! -f "/app/custom_nodes/comfyui-frame-interpolation/ckpts/rife/rife47.pth" ]; then
    echo "Downloading rife47.pth..."
    wget -nc https://github.com/hzwer/Practical-RIFE/releases/download/v4.7/rife47.pth -P /app/custom_nodes/comfyui-frame-interpolation/ckpts/rife/
fi

echo "Ensuring placeholder input files exist..."
mkdir -p /app/input
if [ ! -f "/app/input/avatar.png" ]; then
    wget -nc https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/input/example.png -O /app/input/avatar.png
fi
if [ ! -f "/app/input/avatar2.png" ]; then
    wget -nc https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/input/example.png -O /app/input/avatar2.png
fi
if [ ! -f "/app/input/voice.wav" ]; then
    # Download a small public domain audio sample as placeholder
    wget -nc https://actions.google.com/sounds/v1/alarms/beep_short.ogg -O /app/input/voice.wav
fi

echo "All models and inputs initialized. Starting ComfyUI..."
exec python main.py --listen 0.0.0.0 --port 8188
