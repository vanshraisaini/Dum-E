FROM huggingface/lerobot-gpu

USER root

SHELL [ "/bin/bash", "-c" ]

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git nano \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/vanshraisaini/Dum-E.git

WORKDIR /workspace/Dum-E

COPY datasets/libero_spatial_image/ /workspace/Dum-E/datasets/libero_spatial_image


