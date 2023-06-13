import os
from env import ChopperScape
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from config import storage_root


print("initializing environment")
env = ChopperScape()
env = Monitor(env)
print("training agent")
model = DQN('MlpPolicy', env, buffer_size=10000, learning_rate=1e-3, verbose=1)
# Train the agent
eval_callback = EvalCallback(env, best_model_save_path=f"{storage_root}/models_chopper/",
                             log_path=f"{storage_root}/logs_chopper/", eval_freq=500,
                             deterministic=True, render=True)
model.learn(total_timesteps=int(2e5), callback=eval_callback)
model.save("dqn_chopper")
