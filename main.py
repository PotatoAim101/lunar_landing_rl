from src.env.lunar_lander import LunarLander
from src.env.lunar_lander_multi import LunarLanderMulti
from src.agents.Policy_Gradient_agent import PolicyGradientModel, train_pg
import numpy as np
import torch
from torch.distributions import Categorical

"""
from pyglet.window import key
import matplotlib.pyplot as plt
from src.agents.DDPG_agent import *
import config


def key_press(k, mod):
    if k == 0xFF0D:
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0


def play(env):
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a, total_reward)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def train_ddpg(env, n_games):
    agent = DDPG(input_shape=env.observation_space.shape, env=env,
                 num_actions=env.action_space.shape[0])

    figure_file = config.plots_folder / "ddpg/"
    load_checkpoint = False

    score_history = agent.train(env, load_checkpoint, n_games)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)"""


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
    """global restart

    a = np.array([0.0, 0.0, 0.0])

    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, config.video_folder / "ddpg/", force=True)

    n_games = 101

    train_ddpg(env, n_games)

    env.close()"""
    demo_pg()
