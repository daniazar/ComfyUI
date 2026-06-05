#!/bin/bash
set -e

# Base models directory
MODELS_DIR="/app/models"

echo "Downloading models for ComfyUI... This will take a while!"

# 1. Download InfiniteTalk 1.3B / Wan2.1 models
# Usually goes to custom_nodes/ComfyUI-InfiniteTalk/models or similar, 
# but let's download the base Wan2.1 model if needed.
# For now, we print instructions since exact paths depend on the nodes.
echo "Downloading Wan2.1..."
mkdir -p $MODELS_DIR/checkpoints
# wget -c "https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main/models/..." -P $MODELS_DIR/checkpoints/

echo "Downloading EchoMimic V3 Flash..."
# wget -c "..." -P $MODELS_DIR/checkpoints/

echo "Note: You should use ComfyUI-Manager inside the UI to install any missing models for the workflows."
echo "Done!"
