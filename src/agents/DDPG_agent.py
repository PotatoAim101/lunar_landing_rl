import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, ReLU, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import config
import pickle
import random

class ReplayMemory:
    """
    Buffer to store the states action and terminal flags
    """

    def __init__(self, max_size, num_states, num_actions, batch_size=64):
        self.memory_ctr = 0
        self.memory_size = max_size
        self.batch_size = batch_size

        self.rewards = np.zeros((self.memory_size, 1))
        self.states = np.zeros((self.memory_size, num_states))
        self.actions = np.zeros((self.memory_size, num_actions))
        self.new_states = np.zeros((self.memory_size, num_states))
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

    def sample_buffer(self):
        maxi = min(self.memory_ctr, self.memory_size)

        batch = np.random.choice(maxi, self.batch_size, replace=False)

        a = self.actions[batch]
        r = self.rewards[batch]
        s = self.states[batch]
        n_s = self.new_states[batch]
        t = self.terminal_memory[batch]

        return a, r, s, n_s, t


class OUActionNoise:
    """To implement better exploration by the Actor network, we use noisy perturbations,
    specifically an Ornstein-Uhlenbeck process for generating noise, as described in the
    paper. It samples noise from a correlated normal distribution.
    https://keras.io/examples/rl/ddpg_pendulum/"""
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Critic(keras.Model):
    def __init__(self, name='critic',
                 chkpt_dir=config.model_folder / "ddpg/", continuous=True):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')

        self.fc1_states = Dense(16, activation='relu')
        self.fc2_states = Dense(32, activation='relu')

        self.fc1_actions = Dense(32, activation='relu')

        self.concat = Concatenate()

        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(256, activation="relu")
        self.out = Dense(1, activation=None)

    def call(self, s, a):
        # a_value = self.fc1(tf.concat([s, tf.reshape(a, (a.shape[0], 1))], axis=1))
        """a_value = self.fc1(tf.concat([s, a], axis=1))
        a_value = self.fc2(a_value)
        res = self.out(a_value)
        return res"""
        s = self.fc1_states(s)
        s = self.fc2_states(s)

        a = self.fc1_actions(a)

        c = self.concat([s, a])
        o = self.fc1(s)
        o = self.fc2(o)
        return self.out(o)


class Actor(keras.Model):
    def __init__(self, name='actor', upper_bound=4,
                 chkpt_dir=config.model_folder / "ddpg/", continuous=True):
        super(Actor, self).__init__()
        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')
        self.upper_bound = upper_bound

        if continuous:
            self.fc1 = Dense(256, activation='relu')
            self.fc2 = Dense(256, activation='relu')
            self.mu = Dense(1, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        else:
            #self.fc1 = Dense(256, activation='relu')
            #self.fc2 = Dense(256, activation='relu')
            #self.mu = Dense(1, activation='softmax')  # , kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
            self.fc1 = ReLU(256)
            self.fc2 = ReLU(256)
            self.mu = Dense(1, activation='softmax')

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        mu = self.mu(x)
        return mu * self.upper_bound


def get_actor(num_states, upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = Input(shape=(num_states,))
    out = Dense(256, activation="relu")(inputs)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1, activation="softmax", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states, num_actions):
    # State as input
    state_input = Input(shape=(num_states))
    state_out = Dense(16, activation="relu")(state_input)
    state_out = Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, action_out])

    out = Dense(256, activation="relu")(concat)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


class DDPG:
    def __init__(self, num_actions, num_states, actor_lr=0.001, critic_lr=0.002,
                 env=None, gamma=0.99, max=100, tau=0.005, fc1=400,
                 fc2=300, batchsize=64, noise=0.2, continuous=True):
        self.num_actions = num_actions
        self.batch_size = batchsize
        self.noise = noise
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayMemory(max, num_states, num_actions)
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=self.noise * np.ones(1))
        if continuous:
            self.max_a = env.action_space.high[0]
            self.min_a = env.action_space.low[0]
            self.continuous = True
        else:
            self.max_a = 4
            self.min_a = 1
            self.continuous = False

        """self.actor = Actor(continuous=continuous)
        self.critic = Critic(continuous=continuous)
        self.target_actor = Actor(name='target_actor', continuous=continuous)
        self.target_critic = Critic(name='target_critic', continuous=continuous)"""
        self.actor = get_actor(num_states, 4)
        self.critic = get_critic(num_states, num_actions)
        self.target_actor = get_actor(num_states, 4)
        self.target_actor = get_critic(num_states, num_actions)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)

        self.random_r = 0.15

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # actor weights
        """w = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            w.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(w)
        # critic weights
        w = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            w.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(w)"""
        for(a, b) in zip(self.target_actor.weights, self.actor.weights):
            a.assign(b * tau + a * (1 - tau))

        for(a, b) in zip(self.target_critic.weights, self.critic.weights):
            a.assign(b * tau + a * (1 - tau))

    def remember(self, s, a, r, new_s, finished):
        self.memory.store_transition(s, a, r, new_s, finished)

    def save_model(self):
        print("saving.....")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        print("finished saving")

    def load_model(self):
        print("loading.....")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        print("finished loading")

    def choose_action(self, o, eval=False):
        s = tf.convert_to_tensor([o], dtype=tf.float32)
        # mazimizing exploration by adding random 4%
        a = self.actor(s)

        if not eval:
            a += tf.random.normal(shape=[self.num_actions], mean=0.0, stddev=self.noise)
        a = tf.clip_by_value(a, self.min_a, self.max_a)

        if not self.continuous:
            a = a.numpy()
            a = a.argmax()

        return a

    def learn(self):
        if self.memory.memory_ctr < self.batch_size:
            return
        a, r, s, new_s, finished = self.memory.sample_buffer()
        actions = tf.convert_to_tensor(a, dtype=tf.float32)
        states = tf.convert_to_tensor(s, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_s, dtype=tf.float32)
        rewards = tf.convert_to_tensor(r, dtype=tf.float32)

        # UPDATE RULE FOR CRITIC: loss function
        """with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            y = rewards + self.gamma * self.target_critic(new_states, target_actions)

            critic_value = self.target_critic(states, target_actions)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.trainable_variables))

        # UPDATE RULE FOR ACTOR:
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            critic_value = self.critic(states, new_policy_actions)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient,
                                                 self.actor.trainable_variables))"""

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            new_critic_value = self.target_critic(new_states, target_actions)
            new_critic_value = tf.squeeze(new_critic_value, 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * new_critic_value * (1 - finished)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.trainable_variables))

        # UPDATE RULE FOR ACTOR:
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)  # - for gradient accent for max total score over time
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient,
                                                 self.actor.trainable_variables))

        # update of the target networks
        self.update_network_parameters()

    def play_one_episode(self, env, eval, load_chkpt):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = self.choose_action(observation, eval)
            r = random.random()
            if r <= self.random_r:
                a = random.randint(0, 3)
                self.random_r = self.random_r * 0.9

            if self.continuous:
                observation_, reward, done, info = env.step(action)
            else:
                observation_, reward, done, info = env.step(int(action))
            score += reward
            self.remember(observation, action, reward, observation_, done)
            if not load_chkpt:
                self.learn()
            observation = observation_
        return score

    def train(self, env, load_chkpt, n_games):
        best_score = env.reward_range[0]
        score_history = []
        if load_chkpt:
            n_steps = 0
            while n_steps <= self.batch_size:
                observation = env.reset()
                action = env.action_space.sample()
                observation_, reward, done, info = env.step(action)
                self.remember(observation, action, reward, observation_, done)
                n_steps += 1
            self.learn()
            self.load_model()
            evaluate = True
        else:
            evaluate = False

        for i in range(n_games):
            score = self.play_one_episode(env, evaluate, load_chkpt)

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_chkpt:
                    self.save_model()
                    print("Saving history...")
                    with open(config.plots_folder / "ddpg/history.txt", "wb") as f:
                        pickle.dump(score_history, f)
                    print("History saved")

            print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

        return score_history
