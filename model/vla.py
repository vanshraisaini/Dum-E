import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .vlm_backbone import VLMBackbone
from .action_head import FlowMatchingActionHead


def tile_camera_views(image_tensors: list[torch.Tensor]) -> list[Image.Image]:
    """
    PaliGemma expects one image per sample. For multi-camera setups (e.g. LIBERO's
    agent view + wrist view), the simplest fix without modifying the VLM is to tile
    the views side by side into a single composite image before passing them in.
    If you'd rather keep views separate, swap this for a VLM that natively supports
    multiple images (e.g. Qwen2.5-VL) instead — that's a one-line change to
    vlm_backbone.py's model class.
    """
    tiled = []
    for views in image_tensors:  # views: list of (C, H, W) tensors for one sample
        pil_views = [
            Image.fromarray((v.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
            for v in views
        ]
        widths, heights = zip(*(im.size for im in pil_views))
        composite = Image.new("RGB", (sum(widths), max(heights)))
        x_off = 0
        for im in pil_views:
            composite.paste(im, (x_off, 0))
            x_off += im.width
        tiled.append(composite)
    return tiled


class VLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = VLMBackbone(cfg)
        self.action_head = FlowMatchingActionHead(cfg, self.backbone.hidden_size)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        batch keys expected (straight from a LeRobotDataset DataLoader with
        delta_timestamps set on 'action'):
            batch["observation.images.image"]         (B, C, H, W)
            batch["observation.images.image2"]        (B, C, H, W)
            batch["observation.state"]                (B, state_dim)
            batch["action"]                           (B, chunk, action_dim)
            batch["task"]                             list[str], language instructions
        """
        device = next(self.parameters()).device
        cfg = self.cfg

        images = [
            [batch[k][i] for k in cfg.image_keys] for i in range(batch["action"].shape[0])
        ]
        pil_images = tile_camera_views(images)
        texts = batch["task"]

        vlm_hidden, vlm_mask = self.backbone(pil_images, texts)

        state = batch["observation.state"].to(device)     # (B, state_dim=8)
        actions = batch["action"].to(device)               # (B, chunk, action_dim=7)

        B, T, _ = actions.shape
        noise = torch.randn_like(actions)
        t = torch.rand(B, device=device)  # sample flow timestep per example

        # Rectified-flow-style linear interpolation path between noise and data.
        t_ = t.view(B, 1, 1)
        noisy_actions = (1 - t_) * noise + t_ * actions
        target_velocity = actions - noise  # constant velocity field for linear path

        pred_velocity = self.action_head(
            noisy_actions=noisy_actions,
            timesteps=t,
            state=state,
            vlm_hidden_states=vlm_hidden,
            vlm_attention_mask=vlm_mask,
        )

        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict) -> torch.Tensor:
        """Run Euler integration from noise to produce a clean action chunk."""
        device = next(self.parameters()).device
        cfg = self.cfg

        images = [[observation[k] for k in cfg.image_keys]]
        pil_images = tile_camera_views(images)
        texts = [observation["task"]]

        vlm_hidden, vlm_mask = self.backbone(pil_images, texts)
        state = observation["observation.state"].unsqueeze(0).to(device)  # (1, state_dim=8)

        B = 1
        x = torch.randn(B, cfg.action_chunk_size, cfg.action_dim, device=device)
        dt = 1.0 / cfg.num_inference_steps

        for step in range(cfg.num_inference_steps):
            t = torch.full((B,), step * dt, device=device)
            v = self.action_head(
                noisy_actions=x,
                timesteps=t,
                state=state,
                vlm_hidden_states=vlm_hidden,
                vlm_attention_mask=vlm_mask,
            )
            x = x + dt * v  # Euler step

        return x  # (1, chunk, action_dim)