from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalConfig:

    output_dir: str = "eval_results"
    model_file: str = "checkpoints/checkpoint_epoch2.pt"
    task_ids: List[int] = field(default_factory=lambda: [0])
    task_suite_name: str = "libero_spatial"  # can also choose libero_10, libero_spatial, libero_object, etc.
    action_execution_horizon: int = 30  # number of actions to execute in the environment per action chunk
    max_eval_steps: int = 10  # maximum number of steps to run in the environment per episode