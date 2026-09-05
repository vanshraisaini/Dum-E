import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from .vlm_backbone import VLMBackbone
from .action_head import FlowMatchingActionHead


def to_pil(image_tensor: torch.Tensor, resize: int) -> Image.Image:
    """(C, H, W) float tensor in [0,1] -> resized PIL image."""
    img = TF.resize(image_tensor, [resize, resize], antialias=True)
    arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


class VLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = VLMBackbone(cfg)
        self.action_head = FlowMatchingActionHead(cfg, self.backbone.hidden_size)

    def _prepare_images(self, batch_or_obs: dict, batch_size: int) -> list[list[Image.Image]]:
        """Returns, per sample, a list of resized PIL images (one per camera key)."""
        cfg = self.cfg
        images_per_sample = []
        for i in range(batch_size):
            views = [to_pil(batch_or_obs[k][i], cfg.image_resize) for k in cfg.image_keys]
            images_per_sample.append(views)
        return images_per_sample

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        batch keys expected (straight from a LeRobotDataset DataLoader with
        delta_timestamps set on 'action'):
            batch["observation.images.image"]        (B, C, H, W)
            batch["observation.images.wrist_image"]   (B, C, H, W)
            batch["observation.state"]                (B, state_dim)
            batch["action"]                           (B, chunk, action_dim)
            batch["task"]                             list[str], language instructions
        """
        device = next(self.parameters()).device
        B = batch["action"].shape[0]

        images_per_sample = self._prepare_images(batch, B)
        texts = batch["task"]

        vlm_hidden, vlm_mask = self.backbone(images_per_sample, texts)

        state = batch["observation.state"].to(device)     # (B, state_dim=8)
        actions = batch["action"].to(device)               # (B, chunk, action_dim=7)

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
            vlm_hidden_states=vlm_hidden.float(),
            vlm_attention_mask=vlm_mask,
        )

        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict) -> torch.Tensor:
        """Run Euler integration from noise to produce a clean action chunk."""
        device = next(self.parameters()).device
        cfg = self.cfg

        images_per_sample = self._prepare_images(
            {k: v.unsqueeze(0) for k, v in observation.items() if k in cfg.image_keys}, 1
        )
        texts = [observation["task"]]

        vlm_hidden, vlm_mask = self.backbone(images_per_sample, texts)
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