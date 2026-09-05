FROM huggingface/lerobot-gpu

USER root

SHELL [ "/bin/bash", "-c" ]

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git nano \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace/datasets

COPY datasets/libero_spatial_image/ /workspace/datasets/libero_spatial_image
COPY model/ /workspace/model
COPY scripts/ /workspace/scripts
COPY requirements.txt /workspace/requirements.txt


