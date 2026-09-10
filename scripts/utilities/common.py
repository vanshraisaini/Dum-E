import os

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark
from libero.libero import get_libero_path


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


def get_tasks(cfg):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = cfg.eval.task_suite_name
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Loading task suite: {task_suite_name}!")
    return task_suite

