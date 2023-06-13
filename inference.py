from env import ChopperScape
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from config import model


print("initializing environment")
env = ChopperScape()
env = Monitor(env)
print("training agent")
# model = DQN('MlpPolicy', env, buffer_size=10000, learning_rate=1e-3, verbose=1)

model = DQN.load(model)
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(info)
    env.render()
