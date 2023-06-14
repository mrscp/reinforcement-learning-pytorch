import torch

from env import ChopperScape
from dqn import DQN
from config import storage_root
from itertools import count


if __name__ == '__main__':
    env = ChopperScape()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(env, device, storage_root)
    print("training")
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        env.render()

        if terminated or truncated:
            break

    env.close()
