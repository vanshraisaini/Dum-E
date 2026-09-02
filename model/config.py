from dataclasses import dataclass, field
from typing import List


@dataclass
class VLAConfig:
    # --- VLM backbone ---
    vlm_name_or_path: str = "google/paligemma2-3b-pt-224"
    freeze_vision_encoder: bool = False   # you have 64GB VRAM, full FT is fine to start
    freeze_language_model: bool = False
    vlm_dtype: str = "bfloat16"           # "bfloat16" | "float32"

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
        default_factory=lambda: ["observation.images.image", "observation.images.wrist_image"]
    )