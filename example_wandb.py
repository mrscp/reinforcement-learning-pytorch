import gym
import os
import wandb
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from wandb.integration.sb3 import WandbCallback
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 30000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="intro_to_gym",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 200 == 0, video_length=200)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_CartPole_30k')
model.save(PPO_path)
evaluate_policy(model, env, n_eval_episodes=1000, render=True)
run.finish()
