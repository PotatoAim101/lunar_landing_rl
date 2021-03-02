from car_racing import *
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class ReplayMemory:
    """
    Buffer to store the states action and terminal flags
    """
    def __init__(self, max, input_dim, num_actions):
        self.memory_ctr = 0
        self.memory_size = max
        self.rewards = np.zeros(self.memory_size)
        self.states = np.zeros((self.memory_size, *input_dim))
        self.actions = np.zeros((self.memory_size, num_actions))
        self.new_states = np.zeros((self.memory_size, *input_dim))
        # bool to use it to mask tensors
        # the value of the terminal state is 0 we reset the episode to initial state
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, s, a, r, new_s, finished):
        i = self.memory_ctr % self.memory_size

        self.actions[i] = a
        self.states[i] = s
        self.new_states[i] = new_s
        self.rewards[i] = r
        self.terminal_memory[i] = finished

        self.memory_ctr += 1

    def sample_buffer(self, batch_size):
        max = min(self.memory_ctr, self.memory_size)

        batch = np.random.choice(max, batch_size, replace=False)

        a = self.actions[batch]
        r = self.rewards[batch]
        s = self.states[batch]
        n_s = self.new_states[batch]
        t = self.terminal_memory[batch]

        return a, r, s, n_s, t


class Critic(keras.Model):
    def __init__(self, fc1_dim=512, fc2_dim=512, name='critic',
                 chkpt_dir='models'):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')

        self.fc1 = Dense(fc1_dim, activation='relu')
        self.fc2 = Dense(fc2_dim, activation='relu')
        self.output = Dense(1, activation=None)

    def forward(self, s, a):
        a_value = self.fc1(tf.concat([s, a]), axis=1)
        a_value = self.fc2(a_value)
        res = self.output(a_value)
        return res


class Actor(keras.Model):
    def __init__(self, num_actions, fc1_dim=512, fc2_dim=512, name='actor',
                 chkpt_dir='models'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')

        self.fc1 = Dense(fc1_dim, activation='relu')
        self.fc2 = Dense(fc2_dim, activation='relu')
        self.mu = Dense(num_actions, activation='tanh')

    def forward(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        mu = self.mu(x)
        return mu


class DDPG:
    def __init__(self, num_actions, input_shape, actor_lr=0.001, critic_lr=0.002,
                 env=None, gamma=0.99, max=1000000, tau=0.005, fc1=400,
                 fc2=300, batchsize=64, noise=0.1):
        self.num_actions = num_actions
        self.batch_size = batchsize
        self.noise = noise
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayMemory(max, input_shape, num_actions)

        self.actor = Actor(num_actions=num_actions)
        self.critic = Critic(num_actions=num_actions)
        self.target_actor = Actor(num_actions=num_actions, name='target_actor')
        self.target_critic = Critic(num_actions=num_actions, name='target_critic')

        self.actor.compile(optimize=Adam(learning_rate=actor_lr))
        self.critic.compile(optimize=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimize=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimize=Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # actor weights
        w = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            w.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(w)
        # critic weights
        w = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            w.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(w)

    def remember(self, s, a, r, new_s, finished):
        self.memory.store_transition(s, a, r, new_s, finished)

    def save_model(self):
        print("saving.....")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        print("finished saving")

    def save_model(self):
        print("loading.....")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        print("finished loading")

    def choose_action(self, o, eval=False):
        s = tf.convert_to_tensor([o], dtype=tf.float32)
        a = self.actor(s)
        if not eval:
            a += tf.random.normal(shape=[self.num_actions], mean=0.0, stddev=self.noise)
        a = tf.clip_by_value(a, self.min_a, self.max_a)

        return a[0]

    def learn(self):
        if self.memory.memory_ctr < self.batch_size:
            return
        s, a, r, new_s, finished = self.memory.sample_buffer(self.batch_size)
        actions = tf.convert_to_tensor(a, dtype=tf.float32)
        states = tf.convert_to_tensor(s, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_s, dtype=tf.float32)
        rewards = tf.convert_to_tensor(r, dtype=tf.float32)

        # update rules for critic: loss function
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            new_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = r + self.gamma*new_critic_value*(1-finished)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self)

        #unfinished

    def play_one_episode(self):
        pass

    def train(self, env, a):
        isopen = True
        while isopen:
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                s, r, done, info = env.step(a)
                """total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1"""
                self.play_one_episode()
                isopen = env.render()
                if done or restart or isopen == False:
                    break
