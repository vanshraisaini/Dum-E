"""
Flow-matching action head, following the pi0 / SmolVLA pattern:
- take a noised action chunk + timestep + robot state
- cross-attend to VLM hidden states (vision+language conditioning)
- predict the velocity field that denoises the chunk

At inference, integrate this velocity field over `num_inference_steps` Euler
steps starting from pure noise to produce a clean action chunk.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FlowMatchingActionHead(nn.Module):
    def __init__(self, cfg, vlm_hidden_size: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.action_hidden_dim

        # Project inputs into the action-head's working dimension.
        self.action_in_proj = nn.Linear(cfg.action_dim, d)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.time_embed = SinusoidalTimeEmbedding(d)
        self.vlm_proj = nn.Linear(vlm_hidden_size, d)  # match VLM dim -> action dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.action_num_heads,
            dim_feedforward=cfg.action_ffn_dim,
            dropout=cfg.action_dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.action_num_layers)

        self.action_out_proj = nn.Linear(d, cfg.action_dim)

        # Learned positional embeddings for the action chunk sequence.
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.action_chunk_size, d))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, chunk, action_dim)
        timesteps: torch.Tensor,       # (B,) in [0, 1]
        state: torch.Tensor,           # (B, state_dim)
        vlm_hidden_states: torch.Tensor,  # (B, seq_len, vlm_hidden)
        vlm_attention_mask: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        B, T, _ = noisy_actions.shape

        x = self.action_in_proj(noisy_actions) + self.pos_embed[:, :T]

        t_emb = self.time_embed(timesteps)          # (B, d)
        s_emb = self.state_proj(state)               # (B, d)
        cond = (t_emb + s_emb).unsqueeze(1)           # (B, 1, d), broadcast into sequence
        x = x + cond

        memory = self.vlm_proj(vlm_hidden_states)     # (B, seq_len, d)
        memory_key_padding_mask = ~vlm_attention_mask.bool()  # True = ignore

        out = self.decoder(
            tgt=x,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        velocity = self.action_out_proj(out)           # (B, chunk, action_dim)
        return velocity