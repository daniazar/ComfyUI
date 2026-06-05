FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy ComfyUI code
COPY . .

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sqlalchemy aiohttp tqdm requests pillow scipy numpy

# Install Custom Nodes
WORKDIR /app/custom_nodes

RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git

RUN git clone -b comfyui https://github.com/MeiGen-AI/InfiniteTalk.git ComfyUI-InfiniteTalk && \
    cd ComfyUI-InfiniteTalk && \
    pip install --no-cache-dir -r requirements.txt

# Reset workdir to app
WORKDIR /app

# Optional: You can copy your patches or workflows over the default here
# COPY custom/comfyui_patches /app/custom_nodes/
# COPY custom/comfy_workflows /app/user/default/workflows/

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
