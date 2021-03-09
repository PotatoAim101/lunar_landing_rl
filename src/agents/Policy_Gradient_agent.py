import numpy as np

import torch
from torch.distributions import Categorical
import torch.nn as nn
import gym
import matplotlib.pyplot as plt


class PolicyGradientModel(nn.Module):
    def __init__(self):
        super(PolicyGradientModel, self).__init__()

        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=0)
        return x


def generate_trajectory(agent, env, episode_max_length=-1):
    """Generate a list of States, Actions, Rewards from the policy given by the agent"""

    s = env.reset()

    # Normalize and change shape in order to be compatible with Conv2D layers of the agent
    s = torch.tensor(np.array(s), dtype=torch.float)

    ep_ended = False
    infos = {}

    list_rewards, list_states, list_logprobs = [], [s], []

    t = 0  # Current time

    while (t != episode_max_length) and not ep_ended:
        # Estimate parameters of the normal distribution
        p = agent(s)

        # Define the distribution thanks to torch.distribution.Normal
        m = Categorical(p)

        # Get the action to perform as a sample of the normal distribution defined previously
        a = m.sample().squeeze()

        # Perform a step in the environment (infos is a dict containing useful infos about the current simulation)
        s, r, ep_ended, infos = env.step(a.detach().numpy())
        env.render()

        s = torch.tensor(np.array(s), dtype=torch.float)

        # list_actions.append(a)
        list_rewards.append(r)
        list_states.append(s)
        list_logprobs.append(m.log_prob(a.detach()))

        t += 1

    list_states.pop()  # Remove last state, which we don't need

    return torch.tensor(list_rewards, dtype=torch.float), torch.cat(list_states, 0), torch.stack(list_logprobs), infos


def train_pg(episodes=100, episode_max_length=-1, gamma=0.9):
    env = gym.make('LunarLander-v2')

    # Create the agent model
    pg_model = PolicyGradientModel()

    optim_pg = torch.optim.Adam(pg_model.parameters(), lr=1e-3)
    # optim_value = torch.optim.Adam(value_model.parameters(), lr=1e-4)

    list_scores = []

    for i in range(episodes):
        # Generate a trajectory
        # list_rewards, list_actions, list_states = generate_trajectory(pg_model, env, episode_max_length)
        list_rewards, list_states, list_logprobs, _ = generate_trajectory(pg_model, env, episode_max_length)

        # Compute the cumulated rewards (with a discount factor of gamma)
        list_G = [0.] * len(list_rewards)
        curr_G = 0.
        for t in range(len(list_rewards) - 1, -1, -1):
            curr_G = list_rewards[t] + gamma * curr_G
            list_G[t] = curr_G
        list_G = torch.tensor(list_G)

        loss = - (list_G * list_logprobs).mean()

        optim_pg.zero_grad()
        loss.backward()
        optim_pg.step()

        print("episode = {}, cumulated reward = {}".format(i, sum(list_rewards)))
        list_scores.append(sum(list_rewards))

        if i % 100 == 0:
            np.savetxt('pg_scores.txt', np.array(list_scores))

    np.savetxt('pg_scores.txt', np.array(list_scores))

    plt.plot(range(episodes), list_scores)
    plt.show()

    torch.save(pg_model, "torch_pg_model")

    env.close()


train_pg(episodes=5000, gamma=0.9, episode_max_length=3500)

