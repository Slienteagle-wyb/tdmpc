import time
from pathlib import Path
from gym_pybullet_drones.envs.single_agent_rl import TakeoffAviary
from gym_pybullet_drones.utils.Logger import Logger

__LOGS__ = 'logs'


def test_rollout(cfg):
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env = TakeoffAviary(gui=cfg.env.render,
                        record=False
                        )
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=work_dir,
                    colab=False
                    )
    obs = env.reset()
    start = time.time()
