from car_racing import CarRacing

import numpy as np

import torch
from torch.distributions import Normal, Categorical
import torch.nn as nn
import gym


class PolicyGradientModel(nn.Module):
    def __init__(self):
        super(PolicyGradientModel, self).__init__()
        # self.layer1 = nn.Linear(8, 10)
        # self.layer2 = nn.Linear(10, 10)
        # self.layer3 = nn.Linear(10, 4)

        self.layer3 = nn.Linear(4, 2)

    def forward(self, x):
        # x = torch.relu(self.layer1(x))
        # x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=-1)
        return x


def generate_trajectory(agent, env, episode_max_length=-1):
    """Generate a list of States, Actions, Rewards from the policy given by the agent"""
    s = env.reset()
    ep_ended = False
    list_rewards, list_actions, list_states = [], [], [s]
    t = 0  # Current time
    while (t != episode_max_length) and not ep_ended:
        p = agent(torch.tensor(s, dtype=torch.float))
        m = Categorical(p)
        a = m.sample()
        s, r, ep_ended, _ = env.step(a.detach().numpy())
        env.render()

        list_actions.append(a)
        list_rewards.append(r)
        list_states.append(s)

        t += 1
    list_states.pop()  # Remove last state, which we don't need
    return torch.tensor(list_rewards, dtype=torch.float), torch.tensor(list_actions, dtype=torch.float), torch.tensor(list_states, dtype=torch.float)


def train_pg(episodes=100, episode_max_length=-1, gamma=0.9):
    # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')

    # Create the agent model
    pg_model = PolicyGradientModel()  # 8, 4

    optim_pg = torch.optim.SGD(pg_model.parameters(), lr=1e-2)

    for i in range(episodes):
        list_rewards, list_actions, list_states = generate_trajectory(pg_model, env, episode_max_length)
        list_G = [0.] * len(list_rewards)
        curr_G = 0.
        for t in range(len(list_rewards) - 1, -1, -1):
            curr_G = list_rewards[t] + gamma * curr_G
            list_G[t] = curr_G
        list_G = torch.tensor(list_G)

        optim_pg.zero_grad()
        p = pg_model(list_states)
        m = Categorical(p)
        loss = - (list_G * m.log_prob(list_actions)).mean()
        loss.backward()
        optim_pg.step()
        print("episode = {}, cumulated reward = {}".format(i, sum(list_rewards)))
    env.close()


train_pg(episodes=2000, gamma=0.99)

