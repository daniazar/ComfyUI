# ComfyUI EchoMimic Patches

This directory contains our customized nodes and fixes for the `ComfyUI_EchoMimic` extension. 
These files enable the "Flash Pro" variant to run sequentially in overlapping chunks, significantly reducing VRAM requirements to support consumer GPUs (e.g. 8GB VRAM) without out-of-memory crashes.

## Modified Files:
- `EchoMimic_node.py` — Added support for automatic video length calculation based on audio input.
- `echomimic_v3/infer_flash_pro.py` — Implemented chunked generation loop with overlap blending to bypass VRAM limitations.
- `echomimic_v3/src/utils.py` — Fixed the `get_image_to_video_latent2` function to handle `resize()` issues when processing a list of PIL Images from our overlapping chunks.
- `echomimic_v3/src/wan_transformer3d_audio_2512.py` — Internal audio condition processing enhancements for the Flash model.

## Installation

Run the installation script from PowerShell at the root of the project to clone the original repository into your ComfyUI environment and apply these patches:

```powershell
.\scripts\install_echomimic_v3.ps1 -ComfyUIDir "C:\Users\danie\workspace\Comfy"
```

Then, run the model downloading script to ensure all weights are present:

```powershell
.\scripts\download_models.ps1
```
