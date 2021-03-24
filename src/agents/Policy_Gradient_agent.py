import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import matplotlib.pyplot as plt


class PolicyGradientModel(nn.Module):
    def __init__(self):
        super(PolicyGradientModel, self).__init__()

        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=0)
        return x


def generate_trajectory(agent, env, episode_max_length=-1, render=False):
    """Generate a list of States, Actions, Rewards from the policy given by the agent"""

    s = env.reset()

    # Normalize and change shape in order to be compatible with Conv2D layers of the agent
    s = torch.tensor(np.array(s), dtype=torch.float)

    ep_ended = False

    list_rewards, list_states, list_logprobs = [], [s], []

    t = 0  # Current step

    while (t != episode_max_length) and not ep_ended:
        # Estimate parameters of the normal distribution
        p = agent(s)

        # Define the distribution thanks to torch.distribution.Normal
        m = Categorical(p)

        # Get the action to perform as a sample of the normal distribution defined previously
        a = m.sample().squeeze()

        # Perform a step in the environment (infos is a dict containing useful infos about the current simulation)
        s, r, ep_ended, _ = env.step(a.detach().numpy())

        if render:
            env.render()

        s = torch.tensor(np.array(s), dtype=torch.float)

        # list_actions.append(a)
        list_rewards.append(r)
        list_states.append(s)
        list_logprobs.append(m.log_prob(a.detach()))

        t += 1

    list_states.pop()  # Remove last state, which we don't need

    return torch.tensor(list_rewards, dtype=torch.float), torch.cat(list_states, 0), torch.stack(list_logprobs)


def train_pg(env, pg_model, episodes=100, episode_max_length=-1, gamma=0.9, lr=1e-3, render=False, save_path=None):

    optim_pg = torch.optim.Adam(pg_model.parameters(), lr=lr)

    for i in range(episodes):

        # Generate a trajectory
        list_rewards, list_states, list_logprobs = generate_trajectory(pg_model, env, episode_max_length, render)

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

        print("episode = {}, length = {}, cumulated reward = {}".format(i, len(list_rewards), sum(list_rewards)))

        if save_path is not None and i % 500 == 0:  # Save model every 500 steps
            torch.save(pg_model, save_path)

    plt.show()

    torch.save(pg_model, save_path)

    env.close()
