from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalConfig:

    output_dir: str = "eval_results"
    model_file: str = "checkpoints/checkpoint_epoch2.pt"
    task_ids: List[int] = field(default_factory=lambda: [0])
    task_suite_name: str = "libero_spatial"  # can also choose libero_10, libero_spatial, libero_object, etc.
    action_execution_horizon: int = 20  # number of actions to execute in the environment per action chunk
    max_eval_steps: int = 10  # maximum number of steps to run in the environment per episode

    action_mean: List[float] = field(default_factory=lambda: [
            0.15312488430795423,
            0.13707241597825376,
            -0.15526779033841448,
            -0.005176474488725037,
            -0.011208756940533639,
            -0.02019425420384803,
            0.08423636062840822
        ])
    action_std: List[float] = field(default_factory=lambda: [
            0.4127313215829766,
            0.34726656846708326,
            0.5086747214221364,
            0.03726436053640721,
            0.07244919040999799,
            0.057623360068868625,
            0.9964457753533575
        ])