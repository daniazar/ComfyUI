FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy only requirements first to cache the pip installations
COPY requirements.txt .

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sqlalchemy aiohttp tqdm requests pillow scipy numpy huggingface_hub GitPython gguf mediapipe==0.10.14

# Install Custom Nodes
WORKDIR /app/custom_nodes

RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git

RUN git clone https://github.com/smthemex/ComfyUI_EchoMimic.git && \
    cd ComfyUI_EchoMimic && \
    pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    python install.py

RUN git clone https://github.com/cubiq/ComfyUI_essentials.git

RUN git clone https://github.com/chrisgoringe/cg-use-everywhere.git

RUN git clone https://github.com/city96/ComfyUI-GGUF.git

RUN git clone https://github.com/rgthree/rgthree-comfy.git

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git

# Reset workdir to app
WORKDIR /app

# Apply our custom patches and workflows
# Since the Docker build context is vendor/ComfyUI, we can copy from custom/
COPY custom/comfyui_patches/ComfyUI_EchoMimic/ /app/custom_nodes/ComfyUI_EchoMimic/
COPY custom/comfy_workflows/echomimic_v3_flash_ui.json /app/user/default/workflows/echomimic_v3_flash_ui.json

# Copy initialization script
COPY init_models.sh /app/init_models.sh
RUN chmod +x /app/init_models.sh

# Finally, copy the rest of ComfyUI code (changes most frequently)
COPY . .

EXPOSE 8188

CMD ["/app/init_models.sh"]
