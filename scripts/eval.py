import os
import cv2
import torch
import draccus
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

from config.train_model_config import VLAConfig
from config.eval_config import EvalConfig
from model.vla import VLA

import matplotlib.pyplot as plt


@dataclass
class Config:
    model: VLAConfig = field(default_factory=VLAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


class VideoRecorder:
    def __init__(self, filename, fps=20, frame_size=(256, 256)):
        self.video = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size
        )

    def write_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 0)
        self.video.write(frame)

    def release(self):
        self.video.release()


def get_model(cfg, device):
    # --- Model ---
    model = VLA(cfg.model).to(device)
    model.load_state_dict(torch.load(cfg.eval.model_file, weights_only=True))
    model.eval()
    return model


def get_tasks(cfg):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = cfg.eval.task_suite_name
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Loading task suite: {task_suite_name}!")
    return task_suite


def get_task_data(task_suite, task_id):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} with name {task_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    return task_bddl_file


def setup_env(task_suite, task_id):
    task_bddl_file = get_task_data(task_suite, task_id)
    env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    obs = env.set_init_state(init_states[init_state_id])
    return env, obs


def to_chw_float(img_hwc):
        # Flip Image upside down to match the camera view in the dataset.
        img_hwc = img_hwc[::-1].copy() 

        # (H, W, 3) uint8 -> (3, H, W) float in [0, 1]
        return torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0


def get_model_input(obs, language_instruction, device):
    state = np.concatenate([
        obs["robot0_eef_pos"],            # (3,) x, y, z
        obs["robot0_eef_quat"],           # (4,) rx, ry, rz, rw
        obs["robot0_gripper_qpos"][:1],   # (1,) gripper
    ]).astype(np.float32)

    

    model_input = {
        "observation.images.image": to_chw_float(obs["agentview_image"]),
        "observation.images.wrist_image": to_chw_float(obs["robot0_eye_in_hand_image"]),
        "observation.state": torch.from_numpy(state).to(device),
        "task": language_instruction,
    }
    return model_input


def main(cfg):

    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(cfg, device)    

    task_suite = get_tasks(cfg) 

    task_ids = cfg.eval.task_ids

    for task_id in task_ids:

        video_recorder = VideoRecorder(f"{cfg.eval.output_dir}/libero_{task_id}.mp4", fps=20, frame_size=(256, 256))

        # create and setup the environment
        env, obs = setup_env(task_suite, task_id)
        language_instruction = task_suite.get_task(task_id).language

        video_recorder.write_frame(obs["agentview_image"])

        done = False

        for step in tqdm(range(cfg.eval.max_eval_steps), desc=f"Evaluating task {task_id}"):

            model_input  = get_model_input(obs, language_instruction, device)
            action_chunk = model.predict_action_chunk(model_input)
            action_chunk = action_chunk.squeeze(0)  # (1, chunk, action_dim) -> (chunk, action_dim)

            for i in range(cfg.eval.action_execution_horizon):
                action = action_chunk[i].cpu().numpy()
                unnormalized_action = action * np.array(cfg.eval.action_std) + np.array(cfg.eval.action_mean)

                obs, reward, done, info = env.step(unnormalized_action)
                video_recorder.write_frame(obs["agentview_image"])
                if(done):
                    print(f"Episode finished!")
                    break

            if(done):
                print(f"Episode finished after {step + 1} eval steps.")
                break

        video_recorder.release()
        env.close()


if __name__ == "__main__":
    cfg = draccus.parse(config_class=Config)
    main(cfg)