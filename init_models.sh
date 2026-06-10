#!/bin/bash
set -e

echo "Ensuring model directories exist..."
mkdir -p /app/models/vae
mkdir -p /app/models/clip
mkdir -p /app/models/clip_vision
mkdir -p /app/models/echo_mimic
mkdir -p /app/custom_nodes/comfyui-frame-interpolation/ckpts/rife/

echo "Downloading required models (this will be skipped if they already exist)..."

cat << 'EOF' > download_models.py
import os
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

def dl_file(repo, repo_path, local_name, ldir):
    final_path = os.path.join(ldir, local_name)
    if os.path.exists(final_path):
        print(f"Skipping {local_name}, already exists in {ldir}")
        return
    print(f"Downloading {repo_path} from {repo}")
    dl_path = hf_hub_download(repo_id=repo, filename=repo_path, local_dir=ldir, local_dir_use_symlinks=False)
    if dl_path != final_path and os.path.exists(dl_path):
        shutil.move(dl_path, final_path)

def dl_snap(repo, pattern, ldir):
    if os.path.exists(ldir) and len(os.listdir(ldir)) > 0:
        print(f"Skipping {pattern}, {ldir} is not empty")
        return
    print(f"Downloading {pattern} from {repo}")
    snapshot_download(repo_id=repo, allow_patterns=pattern, local_dir=ldir, local_dir_use_symlinks=False)

dl_file("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "Wan2_1_VAE_bf16.safetensors", "/app/models/vae")
dl_file("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "/app/models/clip")
dl_file("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision_h.safetensors", "/app/models/clip_vision")

dl_snap("BadToBest/EchoMimicV3", ["echomimicv3-flash-pro/*"], "/app/models/echo_mimic")
dl_snap("BadToBest/EchoMimicV3", ["chinese-wav2vec2-base/*"], "/app/models/echo_mimic")
dl_snap("BadToBest/EchoMimicV3", ["transformer/*"], "/app/models/echo_mimic")
EOF

python download_models.py

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
