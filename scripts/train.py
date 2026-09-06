import torch
import wandb
import draccus
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from model.config import VLAConfig
from model.vla import VLA


def get_datasets(cfg):
    print("Loading Datasets!")
    delta_timestamps = {
        "action": [i / 10.0 for i in range(cfg.action_chunk_size)],
    }

    # First load the full dataset just to get episode information
    full_dataset = LeRobotDataset(
        repo_id="lerobot/libero_spatial_image",
        root="datasets/libero_spatial_image",
        delta_timestamps=delta_timestamps,
    )

    num_episodes = full_dataset.num_episodes

    generator = torch.Generator().manual_seed(cfg.seed)

    episode_indices = torch.randperm(
        num_episodes,
        generator=generator
    ).tolist()

    num_val = int(num_episodes * cfg.val_split)

    val_episodes = episode_indices[:num_val]
    train_episodes = episode_indices[num_val:]

    print(f"Total episodes: {num_episodes}")
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes:   {len(val_episodes)}")

    train_dataset = LeRobotDataset(
        repo_id="lerobot/libero_spatial_image",
        root="datasets/libero_spatial_image",
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
    )

    val_dataset = LeRobotDataset(
        repo_id="lerobot/libero_spatial_image",
        root="datasets/libero_spatial_image",
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
    )

    print("Datasets loaded!")
    return train_dataset, val_dataset


def main(cfg):

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.__dict__,
        )

    train_dataset, val_dataset = get_datasets(cfg)
    
    loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.train_data_loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train_batch_size, shuffle=False, num_workers=4)

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLA(cfg).to(device)
    model.train()

    # --- Optimizer and Scheduler ---
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * len(loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        #Model Training
        for step, batch in enumerate(loader):
            
            loss = model.compute_loss(batch)
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            global_step += 1

            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if cfg.use_wandb and step % cfg.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch,
                        "train/lr": optim.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        #Saving Checkpoint
        ckpt_path = f"checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if cfg.use_wandb:
            wandb.save(ckpt_path)  # uploads checkpoint alongside the run

        # Model Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.inference_mode():
            for batch in val_loader:
                loss = model.compute_loss(batch)
                val_loss += loss.item()
                num_val_batches += 1
        val_loss /= num_val_batches

        print(
            f"epoch {epoch} "
            f"val_loss={val_loss:.4f}"
        )

        if cfg.use_wandb:
            wandb.log(
                {
                    "val/loss": val_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )

        del loss, batch, val_loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg = draccus.parse(config_class=VLAConfig)
    main(cfg)