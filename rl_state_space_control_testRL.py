import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import argparse

from environments import *

from algorithms import PI

from stable_baselines3 import DDPG

#Set WandB project
wandb.init(
    project="PMSM-RL",  # Project name in WandB
    name="DDPG_Training",  # This run's name
    sync_tensorboard=False,  #Stop syncing tensorboard
    monitor_gym=True,  # Monitor Gym environment
    save_code=True,
)
log_dir = "wandb_logs/"
os.makedirs(log_dir, exist_ok=True)

# CLI Input
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="PMSM",
                    choices=['LoadRL', 'Load3RL', 'PMSM'], help='Environment name')
parser.add_argument("--reward_function", type=str, default="quadratic",
                    choices=['absolute', 'quadratic', 'quadratic_2', 'square_root', 'square_root_2',
                             'quartic_root', 'quartic_root_2'], help='Reward function type')
parser.add_argument("--job_id", type=str, default="")
parser.add_argument("--train", action="store_true", help="Enable training")
parser.add_argument("--test", action="store_true", help="Enable testing")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")

args = parser.parse_args()
env_name        = args.env_name
reward_function = args.reward_function
job_id          = args.job_id
train           = args.train
test            = args.test
epochs         = args.epochs
batch_size     = args.batch_size
gamma          = args.gamma

if env_name == "LoadRL":
    if reward_function in ["quadratic_2", "square_root_2", "quartic_root_2"]:
        sys.exit("This reward function has not been implemented for this environment")
    sys_params_dict = {"dt": 1 / 10e3,  # Sampling time [s]
                       "r": 1,          # Resistance [Ohm]
                       "l": 1e-2,       # Inductance [H]
                       "vdc": 500,      # DC bus voltage [V]
                       "i_max": 100,    # Maximum current [A]
                       }
elif env_name == "Load3RL":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 1,              # Resistance [Ohm]
                       "l": 1e-2,           # Inductance [H]
                       "vdc": 500,          # DC bus voltage [V]
                       "we_nom": 200*2*np.pi, # Nominal speed [rad/s]
                       }
    idq_max_norm = lambda vdq_max,we,r,l: vdq_max / np.sqrt(np.power(r, 2) + np.power(we * l, 2))
    # Maximum current [A]
    sys_params_dict["i_max"] = idq_max_norm(sys_params_dict["vdc"]/2, sys_params_dict["we_nom"],
                                            sys_params_dict["r"], sys_params_dict["l"])
elif env_name == "PMSM":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "ld": 0.91e-3,       # Inductance d-frame [H]
                       "lq": 1.17e-3,       # Inductance q-frame [H]
                       "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 200,            # Maximum current [A]
                       }
else:
    raise NotImplementedError
    # sys.exit("Environment name not existant")

environments = {"LoadRL": {"env": EnvLoadRL,
                    "name": f"Single Phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 200,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoadRL_{reward_function}"
                    },
                "Load3RL": {"env": EnvLoad3RL,
                    "name": f"Three-phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoad3RL_{reward_function}"
                    },
                "PMSM": {"env": EnvPMSM,
                    "name": f"PMSM / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 6000,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvPMSM_{reward_function}"
                    },
                }

env_sel = environments[env_name]                # Choose Environment


sys_params_dict["reward"] = env_sel["reward"] 
env = env_sel["env"](sys_params=sys_params_dict)
# env = gym.wrappers.TimeLimit(env, env_sel["max_episode_steps"])
# env = gym.wrappers.RecordEpisodeStatistics(env)


# Set up environment
def make_env():
    env_instance = env_sel["env"](sys_params=sys_params_dict)
    env_instance = gym.wrappers.TimeLimit(env_instance, max_episode_steps=env_sel["max_episode_steps"])
    env_instance = Monitor(env_instance, log_dir)
    return env_instance

env = DummyVecEnv([make_env])  #Vectorized environment
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


if train:
    print(f"Model: {env_sel['model_name']}, Reward: {env_sel['reward']}, Epochs: {epochs}, Batch Size: {batch_size}, Gamma: {gamma}")
    
    os.makedirs("models", exist_ok=True)
    
    model = DDPG("MlpPolicy", env, batch_size=batch_size, gamma=gamma, verbose=1)
    
    for epoch in range(epochs):
        obs = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Choose optimal action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1  # 记录步数
            
            
            wandb.log({
                "epoch": epoch+1,
                "step_in_epoch": step_count,
                "reward": reward,
                "total_reward": total_reward,
                "id": obs[0,0],
                "iq": obs[0,1],
                "vd": action[0,0],
                "vq": action[0,1],
            })
        
        print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}")

    model.save(f"models/{env_sel['model_name']}")
    env.close()
    sys.exit()



