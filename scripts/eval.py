import os
import torch
import draccus
import numpy as np
import math
from tqdm import tqdm
from dataclasses import dataclass, field

from config.train_model_config import VLAConfig
from config.eval_config import EvalConfig
from model.vla import VLA
from scripts.utilities.video_recorder import VideoRecorder
from scripts.utilities.common import setup_env, get_tasks


@dataclass
class Config:
    model: VLAConfig = field(default_factory=VLAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_model(cfg, device):
    # --- Model ---
    model = VLA(cfg.model).to(device)
    model.load_state_dict(torch.load(cfg.eval.model_file, weights_only=True))
    model.eval()
    return model


def to_chw_float(img_hwc):
        # Flip Image upside down to match the camera view in the dataset.
        img_hwc = img_hwc[::-1].copy() 

        # (H, W, 3) uint8 -> (3, H, W) float in [0, 1]
        return torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0

def quat2axisangle(quat):
    # robosuite/LIBERO convention: quat = (x, y, z, w)
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_model_input(obs, language_instruction, device):
    state = np.concatenate([
        obs["robot0_eef_pos"],                            # (3,) x, y, z
        quat2axisangle(obs["robot0_eef_quat"]),           # (3,) rx, ry, rz 
        obs["robot0_gripper_qpos"],                       # (2,) gripper
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

    success_count = 0

    for task_id in task_ids:

        video_recorder = VideoRecorder(f"{cfg.eval.output_dir}/libero_{task_id}.mp4", fps=20, frame_size=(256, 256))

        # create and setup the environment
        env, obs = setup_env(task_suite, task_id)
        language_instruction = task_suite.get_task(task_id).language

        video_recorder.write_frame(obs["agentview_image"], language_instruction)

        done = False

        for step in tqdm(range(cfg.eval.max_eval_steps), desc=f"Evaluating task {task_id}"):

            model_input  = get_model_input(obs, language_instruction, device)
            action_chunk = model.predict_action_chunk(model_input)
            action_chunk = action_chunk.squeeze(0)  # (1, chunk, action_dim) -> (chunk, action_dim)

            for i in range(cfg.eval.action_execution_horizon):
                action = action_chunk[i].cpu().numpy()

                obs, reward, done, info = env.step(action)
                video_recorder.write_frame(obs["agentview_image"], language_instruction)
                if(done):
                    break

            if(done):
                print(f"Episode finished after {step + 1} eval steps.")
                success_count+=1
                break

        video_recorder.release()
        env.close()

    print("Evaluation complete!")
    print("Total Episodes = ", len(task_ids))
    print(f"Successful Epiosodes = {success_count}")
    print("Success Rate = ", (success_count/len(task_ids))*100, "%")


if __name__ == "__main__":
    cfg = draccus.parse(config_class=Config)
    main(cfg)