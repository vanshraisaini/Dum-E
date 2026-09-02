"""
Wraps a pretrained SmolVLM2 model and exposes its per-token hidden states so
the action head can cross-attend to them.

Why SmolVLM2 at this VRAM budget: at 256M-500M params, full fine-tuning
(weights + grads + optimizer states + activations) fits comfortably in 8GB,
SmolVLM2 is also what SmolVLA itself is built on, so this is a reasonable
lineage to follow.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText


class VLMBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.vlm_dtype == "bfloat16" else torch.float32

        self.processor = AutoProcessor.from_pretrained(cfg.vlm_name_or_path)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            cfg.vlm_name_or_path, torch_dtype=dtype
        )

        if cfg.gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable()

        if cfg.freeze_vision_encoder:
            for p in self.vlm.model.vision_model.parameters():
                p.requires_grad = False
        if cfg.freeze_language_model:
            for p in self.vlm.model.text_model.parameters():
                p.requires_grad = False

        # SmolVLM2's text tower hidden size (works the same way across the family).
        self.hidden_size = self.vlm.config.text_config.hidden_size

    def _build_conversations(self, images_per_sample: list[list], texts: list[str]) -> list[list[dict]]:
        """
        images_per_sample: list of length B, each a list of PIL images (one per
                            camera view) for that sample.
        texts:              list of length B, one language instruction per sample.
        """
        conversations = []
        for views, instruction in zip(images_per_sample, texts):
            content = [{"type": "image", "image": img} for img in views]
            content.append({"type": "text", "text": instruction})
            conversations.append([{"role": "user", "content": content}])
        return conversations

    def forward(self, images_per_sample: list[list], texts: list[str]):
        """
        Returns:
            hidden_states: (B, seq_len, hidden_size) — per-token features to
                            condition the action head on.
            attention_mask: (B, seq_len)
        """
        conversations = self._build_conversations(images_per_sample, texts)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        outputs = self.vlm.model(  # base model (not the generation head) — cheaper, gives hidden states
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return hidden_states, attention_mask