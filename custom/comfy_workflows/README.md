# InfiniteTalk ComfyUI Workflows

Place the InfiniteTalk workflow JSON files here. Download them from:
https://huggingface.co/MeiGen-AI/InfiniteTalk/tree/main/comfyui

## Required Files

> [!NOTE]
> For a detailed breakdown of which model to use based on your machine's capabilities (including M4 Mac Mini vs. NVIDIA GPUs), please review the [Hardware Performance Comparison](../docs/hardware_performance_comparison.md) document.

| Filename | Description | Model |
|---|---|---|
| `infinitetalk_1.3b_i2v.json` | Single person, image-to-video, 1.3B model | Fast, lower quality |
| `infinitetalk_14b_i2v.json` | Single person, image-to-video, 14B model | Slow, high quality |

## Download Steps

1. Go to https://huggingface.co/MeiGen-AI/InfiniteTalk/tree/main/comfyui
2. Download the single-person I2V workflow JSON file
3. Save it here as `infinitetalk_1.3b_i2v.json` (for the 1.3B model)
4. Once you have the 14B weights, save the 14B workflow as `infinitetalk_14b_i2v.json`

## ComfyUI Setup (macOS M1/M2/M3/M4)

> [!IMPORTANT]
> **Python 3.10 or newer is REQUIRED.** Standard macOS Python (3.9.x) will cause errors. 
> We recommend using Homebrew: `brew install python@3.11`

```bash
# 1. Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI

# 2. Create venv with Python 3.10+
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Requirements (Standard + Missing macOS deps)
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install sqlalchemy aiohttp tqdm requests pillow scipy numpy

# 4. Install Custom Nodes
cd custom_nodes
git clone https://github.com/Comfy-Org/ComfyUI-Manager.git
git clone -b comfyui https://github.com/MeiGen-AI/InfiniteTalk.git ComfyUI-InfiniteTalk
cd ComfyUI-InfiniteTalk
pip install -r requirements.txt

# 5. Start Server
cd ~/ComfyUI
source venv/bin/activate
python main.py --port 8188
```

## Environment Variables

Add to your `.env.local`:
```
COMFYUI_URL=http://localhost:8188
NEXT_PUBLIC_COMFYUI_URL=http://localhost:8188
```
