from dataclasses import dataclass, field
from typing import List


@dataclass
class VLAConfig:
    # --- VLM backbone ---
    vlm_name_or_path: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    freeze_vision_encoder: bool = False   # 500M fits full fine-tuning in 8GB; see README
    freeze_language_model: bool = False
    vlm_dtype: str = "bfloat16"           # "bfloat16" | "float32"
    gradient_checkpointing: bool = True   # needed to stay inside 8GB

    # --- Robot / action spec (LIBERO) ---
    state_dim: int = 8
    action_dim: int = 7
    action_chunk_size: int = 50           # how many future steps to predict per chunk

    # --- Action head (flow matching transformer) ---
    action_hidden_dim: int = 1024
    action_num_layers: int = 6
    action_num_heads: int = 8
    action_ffn_dim: int = 4096
    action_dropout: float = 0.0

    # --- Flow matching ---
    num_inference_steps: int = 10         # Euler integration steps at inference time
    flow_sig_min: float = 1e-3            # matches pi0/SmolVLA-style flow matching schedule

    # --- Cameras (must match your LeRobotDataset feature keys) ---
    image_keys: List[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )

    # --- Memory knobs for 8GB VRAM ---
    image_resize: int = 224               # downscale camera frames before the processor
    train_batch_size: int = 1
    grad_accum_steps: int = 8             # effective batch size = train_batch_size * grad_accum_steps

    # --- Logging ---
    use_wandb: bool = True
    wandb_project: str = "dumE-vla"
    wandb_run_name: str | None = None     # None -> wandb auto-generates one
    log_every: int = 10                   # steps between wandb scalar logs
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5