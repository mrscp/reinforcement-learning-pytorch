from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
import math
import random
import matplotlib.pyplot as plt

from env import ChopperScape
from config import storage_root
from models import ModelTorch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        state = state.to("cuda:1")
        action = action.to("cuda:1")
        if next_state is not None:
            next_state = next_state.to("cuda:1")
        reward = reward.to("cuda:1")
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for i in range(len(batch)):
            state = batch[i].state.to("cuda:0")
            action = batch[i].action.to("cuda:0")
            if batch[i].next_state is not None:
                next_state = batch[i].next_state.to("cuda:0")
            else:
                next_state = batch[i].next_state
            reward = batch[i].reward.to("cuda:0")
            batch[i] = Transition(state, action, next_state, reward)
        return batch

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(
        self, env, device, saving_path, lr=1e-4, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000,
        tau=0.005
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.env = env
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.saving_path = saving_path

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state = self.env.reset()
        n_observations = len(state)
        print("state shape", state.shape)
        print("actions", n_actions)
        print("n_observations", n_observations)

        self.policy_net = ModelTorch(state.shape, n_actions).to(device)
        self.target_net = ModelTorch(state.shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(5000)
        self.steps_done = 0
        self.episode_durations = []

    def save(self):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.saving_path + "/model.zip")

    def load(self):
        checkpoint = torch.load(self.saving_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def train(self, n_episodes):
        best_episode = 0
        best_steps = 0
        for i_episode in range(n_episodes):
            print(f"episode {i_episode}")
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                self.env.render()
                if done:
                    if best_steps < t:
                        best_steps = t
                        best_episode = i_episode

                    self.save()
                    print(f"best ep {best_episode}")
                    print(f"memory {len(self.memory)}")
                    print(f"t {t}")
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, batch.next_state)
        ), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


if __name__ == '__main__':
    environ = ChopperScape()
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {d}")

    model = DQN(environ, d, storage_root)
    print("training")
    model.train(num_episodes)

    environ.close()
