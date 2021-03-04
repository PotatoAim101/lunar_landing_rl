from car_racing_new import CarRacing

import numpy as np

import torch
from torch.distributions import Normal, Categorical
import torch.nn as nn
import gym


SIGMA_TURN = 0.1
SIGMA_GAS = 0.1
SIGMA_BRAKE = 0.1


def inv_sigmoid(x):
    return -torch.log((1 / x) - 1)


class PolicyGradientModel(nn.Module):
    def __init__(self):
        super(PolicyGradientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)  # Conv layer with one channel (blue pixels), 4 output channel, size of kernel 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.fc1 = nn.Linear(7744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # Dim of output is 3

    def forward(self, x):
        x = x[:,1:2,:,:] # Only select blue channel
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 7744)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ValueModel(nn.Module):
    def __init__(self):
        super(ValueModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # Dim of output is 1 (the estimate Value of the state)

    def forward(self, x):
        x = x[:, 1:2, :, :]  # Only select blue channel
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_trajectory(agent, env, episode_max_length=-1):
    """Generate a list of States, Actions, Rewards from the policy given by the agent"""

    s = env.reset()

    # Normalize and change shape in order to be compatible with Conv2D layers of the agent
    s = torch.tensor(np.array(s), dtype=torch.float).permute(2, 0, 1).unsqueeze(0) / 255.

    ep_ended = False
    list_rewards, list_actions, list_states = [], [], [s]
    t = 0  # Current time

    while (t != episode_max_length) and not ep_ended:
        # Estimate parameters of the normal distribution
        p = agent(torch.tensor(np.array(s), dtype=torch.float))

        # Define the distribution thanks to torch.distribution.Normal
        m = Normal(p, scale=torch.tensor([SIGMA_TURN, SIGMA_GAS, SIGMA_BRAKE]))

        # Get the action to perform as a sample of the normal distribution defined previously
        a = m.sample().squeeze()

        # Map the action from R to the right domain (two possibilities: sigmoid or clipping)
        a_effective = (torch.sigmoid(a) - torch.tensor([0.5, 0., 0.])) * torch.tensor([2., 1., 0.2])
        # a_effective = torch.max(torch.min(a, torch.tensor([1., 1., 0.1])), torch.tensor([-1., 0., 0.]))

        # Perform a step in the environment
        s, r, ep_ended, _ = env.step(a_effective.detach().numpy())
        s = torch.tensor(np.array(s), dtype=torch.float).permute(2, 0, 1).unsqueeze(0) / 255.
        env.render()

        list_actions.append(a)
        list_rewards.append(r)
        list_states.append(s)
        t += 1

    list_states.pop()  # Remove last state, which we don't need

    return torch.tensor(list_rewards, dtype=torch.float), torch.stack(list_actions), torch.cat(list_states, 0)


def train_pg(episodes=100, episode_max_length=-1, gamma=0.9):

    env = CarRacing()

    # Create the agent model
    pg_model = PolicyGradientModel()

    # Create the value model, to compute baseline
    value_model = ValueModel()

    optim_pg = torch.optim.Adam(pg_model.parameters(), lr=1e-4)
    optim_value = torch.optim.Adam(value_model.parameters(), lr=1e-4)

    for i in range(episodes):
        # Generate a trajectory
        list_rewards, list_actions, list_states = generate_trajectory(pg_model, env, episode_max_length)

        # Compute the cumulated rewards (with a discount factor of gamma)
        list_G = [0.] * len(list_rewards)
        curr_G = 0.
        for t in range(len(list_rewards) - 1, -1, -1):
            curr_G = list_rewards[t] + gamma * curr_G
            list_G[t] = curr_G
        list_G = torch.tensor(list_G)

        optim_pg.zero_grad()

        # Estimate parameters of the normal distribution for each state encountered in the generated trajectory
        p = pg_model(list_states)

        # Define the corresponding distributions
        m = Normal(p, scale=torch.tensor([SIGMA_TURN, SIGMA_GAS, SIGMA_BRAKE]))

        # Compute the baseline value (to reduce variance)
        baseline = value_model(list_states)

        # Compute loss and perform gradient descent on agent model
        loss = - ((list_G - baseline) * m.log_prob(list_actions).sum(axis=1)).sum() / episode_max_length
        loss.backward()
        optim_pg.step()

        # Perform gradient descent on value model
        optim_value.zero_grad()
        baseline = value_model(list_states)
        value_loss = ((list_G - baseline)**2).mean()
        value_loss.backward()
        optim_value.step()

        print("episode = {}, cumulated reward = {} / {}".format(i, sum(list_rewards), value_loss))

    env.close()


train_pg(episodes=20000, gamma=0.8, episode_max_length=3500)

