"""
Wraps a pretrained VLM(PaliGemma2-style) and exposes its per-token hidden states
so the action head can cross-attend to them.

Why PaliGemma2: it's SigLIP (vision) + Gemma (language) already fused and
pretrained together, which is what pi0 and most current VLAs build on. You get
a working vision+language encoder in one call instead of hand-wiring SigLIP and
a separate LM together.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class VLMBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.vlm_dtype == "bfloat16" else torch.float32

        self.processor = AutoProcessor.from_pretrained(cfg.vlm_name_or_path)
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.vlm_name_or_path, torch_dtype=dtype
        )

        # Freezing options — useful if you OOM or want faster early iteration.
        if cfg.freeze_vision_encoder:
            for p in self.vlm.vision_tower.parameters():
                p.requires_grad = False
        if cfg.freeze_language_model:
            for p in self.vlm.language_model.parameters():
                p.requires_grad = False

        self.hidden_size = self.vlm.config.text_config.hidden_size

    def forward(self, images: list[torch.Tensor], texts: list[str]):
        """
        images: list of PIL images or pre-batched pixel tensors, one instruction's
                worth of camera views concatenated (PaliGemma takes a single image
                stream; multi-camera handling is done by tiling — see note below).
        texts:  list[str], one language instruction per batch item.

        Returns:
            hidden_states: (B, seq_len, hidden_size) — per-token features to
                            condition the action head on.
            attention_mask: (B, seq_len)
        """
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device, dtype=self.vlm.dtype if hasattr(self.vlm, "dtype") else None)

        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # Last layer hidden states from the language model tower.
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        return hidden_states, attention_mask