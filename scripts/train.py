import torch
import wandb
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from model.config import VLAConfig
from model.vla import VLA


def main():
    cfg = VLAConfig()

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.__dict__,
        )

    # --- Dataset: matches what you already have working locally ---
    delta_timestamps = {
        "action": [i / 10.0 for i in range(cfg.action_chunk_size)],  # fps=10 for LIBERO
    }
    dataset = LeRobotDataset(
        repo_id="lerobot/libero",
        root="datasets/libero",
        delta_timestamps=delta_timestamps,
    )
    loader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2)

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLA(cfg).to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    accum = cfg.grad_accum_steps
    global_step = 0

    for epoch in range(cfg.num_epochs):
        optim.zero_grad()
        for step, batch in enumerate(loader):
            loss = model.compute_loss(batch) / accum
            loss.backward()

            did_optim_step = (step + 1) % accum == 0
            if did_optim_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()
                global_step += 1

            full_loss = loss.item() * accum

            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {full_loss:.4f}")

            if cfg.use_wandb and did_optim_step:
                wandb.log(
                    {
                        "train/loss": full_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch,
                        "train/lr": optim.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        ckpt_path = f"checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if cfg.use_wandb:
            wandb.save(ckpt_path)  # uploads checkpoint alongside the run

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()