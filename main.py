from src.env.car_racing import *
from pyglet.window import key
import matplotlib.pyplot as plt
from src.agents.DDPG_agent import *
from src.agents.DDPG_official import *
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

def ddpg_official(env=None):
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


def train_ddpg(env, n_games):
    agent = DDPG(input_shape=env.observation_space.shape, env=env,
                 num_actions=env.action_space.shape[0])

    figure_file = config.plots_folder / "ddpg/"
    load_checkpoint = False

    score_history = agent.train(env, load_checkpoint, n_games)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)



if __name__ == "__main__":
    global restart

    a = np.array([0.0, 0.0, 0.0])

    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, config.video_folder / "ddpg/", force=True)

    n_games = 101

    train_ddpg(env, n_games)

    env.close()
