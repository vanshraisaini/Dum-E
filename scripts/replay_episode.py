import draccus
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.utilities.video_recorder import VideoRecorder
from dataclasses import dataclass, field
from scripts.utilities.common import setup_env, get_tasks
from tqdm import tqdm

from config.train_model_config import VLAConfig
from config.eval_config import EvalConfig


@dataclass
class Config:
    model: VLAConfig = field(default_factory=VLAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def main(cfg):

    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    task_id = 1
    dataset = LeRobotDataset(
            repo_id="lerobot/libero_spatial_image",
            root="datasets/libero_spatial_image",
        )
    episode = dataset.hf_dataset.filter(lambda x: x["episode_index"] == task_id)

    task_suite = get_tasks(cfg)

    env, obs = setup_env(task_suite, task_id=task_id)  # match episode 0's real task_id
    recorder = VideoRecorder(f"{cfg.eval.output_dir}/libero_task_{task_id}_ground_truth.mp4", fps=20, frame_size=(256, 256))

    for step in tqdm(episode):
        action = step["action"].numpy()  # raw dataset action, no model
        obs, reward, done, info = env.step(action)
        recorder.write_frame(obs["agentview_image"])
        if done:
            print("ground-truth replay succeeded")
            break

    recorder.release()
    env.close()


if __name__ == "__main__":
    cfg = draccus.parse(config_class=Config)
    main(cfg)