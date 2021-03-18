import numpy as np

from src.env.lunar_landing_multi import LunarLander
# from lunar_landing import LunarLander

N = 10
env = LunarLander(N=N)
episode_max_length=3500


for i in range(500):
    s = env.reset()
    env.render()
    each_ep_ended = False
    t = 0  # Current time
    while (t != episode_max_length) and not (each_ep_ended):
        each_ep_ended = True
        for n in range(N):
            a = np.random.randint(0, 4)
            s, r, ep_ended, infos = env.step(a, n=n)
            each_ep_ended = each_ep_ended and ep_ended
        env.render()
        t += 1