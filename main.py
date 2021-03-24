from src.env.lunar_lander import LunarLander
from src.env.lunar_lander_multi import LunarLanderMulti
from src.agents.Policy_Gradient_agent import PolicyGradientModel, train_pg
import numpy as np
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from src.agents.DDPG_agent import *
from src.agents.DDPG_official import *
from src.env.lunar_lander import *


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def train_ddpg():
    env = LunarLander()
    env.render()
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, config.video_folder / "ddpg/", force=True)

    agent = DDPG(env=env, num_actions=4, input_shape=env.observation_space.shape[0],
                 continuous=False)  # env.action_space.shape[0])
    agent.load_model()
    n_games = 51

    figure_file = config.plots_folder / "ddpg/lunar_planer.png"
    load_checkpoint = True

    score_history = agent.train(env, n_games, load_checkpoint)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)


def train_ddpg_official():
    env = LunarLanderContinuous()
    # env = LunarLander()
    env.render()

    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, config.video_folder / "ddpg/", force=True)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    if env.continuous:
        num_actions = env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(num_actions))

        upper_bound = env.action_space.high[0]
        lower_bound = env.action_space.low[0]

        print("Max Value of Action ->  {}".format(upper_bound))
        print("Min Value of Action ->  {}".format(lower_bound))
    else:
        num_actions = env.action_space.n
        print("Size of Action Space ->  {}".format(num_actions))

        upper_bound = num_actions
        lower_bound = 0
        print("Max Value of Action ->  {}".format(upper_bound))
        print("Min Value of Action ->  {}".format(lower_bound))

    ddpg = DDPG_OFF(num_states, num_actions, lower_bound, upper_bound)
    avg_reward_list = ddpg.train(env, 10)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


def pg_play_multiple_models(env, models, episodes=10, episode_max_length=3500):

    N = len(models)

    for _ in range(episodes):
        s = env.reset()

        list_s = [s for _ in range(N)]
        eps_ended = [False] * N
        each_ep_ended = False

        t = 0  # Current step

        while (t != episode_max_length) and not each_ep_ended:
            each_ep_ended = True

            for n in range(N):
                if not eps_ended[n]:
                    s = list_s[n]
                    s = torch.tensor(np.array(s), dtype=torch.float)
                    p = models[n](s)
                    m = Categorical(p)
                    a = m.sample().squeeze()
                    list_s[n], _, eps_ended[n], _ = env.step(a.detach().numpy(), n=n)
                    each_ep_ended = each_ep_ended and eps_ended[n]

            env.render()

            t += 1

    env.close()


def demo_pg():

    # Begins the training of a new Policy Gradient Model on a few episodes
    """env = LunarLander()
    pg_model = PolicyGradientModel()
    train_pg(env, pg_model, episodes=50, episode_max_length=3500, render=True, save_path="models/pg_sample_model")"""

    # Visualize behavior of several models
    env_multi = LunarLanderMulti(N=3)
    model1 = torch.load("models/pg_model_full_train")
    model2 = torch.load("models/pg_model_partially_train")
    model3 = torch.load("models/pg_sample_model")
    pg_play_multiple_models(env_multi, [model1, model2, model3])


if __name__ == "__main__":
    demo_pg()
