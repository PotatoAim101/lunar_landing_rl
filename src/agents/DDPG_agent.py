import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.optimizers import Adam
import config
import pickle


class ReplayMemory:
    """
    Buffer to store the states action and terminal flags
    """

    def __init__(self, max_size, input_dim, num_actions):
        self.memory_ctr = 0
        self.memory_size = max_size
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
        maxi = min(self.memory_ctr, self.memory_size)

        batch = np.random.choice(maxi, batch_size, replace=False)

        a = self.actions[batch]
        r = self.rewards[batch]
        s = self.states[batch]
        n_s = self.new_states[batch]
        t = self.terminal_memory[batch]

        return a, r, s, n_s, t


class Critic(keras.Model):
    def __init__(self, fc1_dim=512, fc2_dim=512, name='critic',
                 chkpt_dir=config.model_folder / "ddpg/", continuous=True):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')

        if continuous:
            self.fc1 = Dense(fc1_dim, activation='relu')
            self.fc2 = Dense(fc2_dim, activation='relu')
        else:
            self.fc1 = ReLU(fc1_dim)
            self.fc2 = ReLU(fc2_dim)
        self.out = Dense(1, activation=None)

    def call(self, s, a):
        # a_value = self.fc1(tf.concat([s, tf.reshape(a, (a.shape[0], 1))], axis=1))
        a_value = self.fc1(tf.concat([s, a], axis=1))
        a_value = self.fc2(a_value)
        res = self.out(a_value)
        return res


class Actor(keras.Model):
    def __init__(self, num_actions, fc1_dim=512, fc2_dim=512, name='actor',
                 chkpt_dir=config.model_folder / "ddpg/", continuous=True):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint_path = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_path, self.model_name + '.h5')

        if continuous:
            self.fc1 = Dense(fc1_dim, activation='relu')
            self.fc2 = Dense(fc2_dim, activation='relu')
            self.mu = Dense(num_actions, activation='tanh')
        else:
            self.fc1 = ReLU(fc1_dim)
            self.fc2 = ReLU(fc2_dim)
            self.mu = Dense(num_actions, activation='softmax')

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        mu = self.mu(x)
        return mu


class DDPG:
    def __init__(self, num_actions, input_shape, actor_lr=0.001, critic_lr=0.002,
                 env=None, gamma=0.99, max=1000000, tau=0.005, fc1=400,
                 fc2=300, batchsize=64, noise=0.1, continuous=True):
        self.num_actions = num_actions
        self.batch_size = batchsize
        self.noise = noise
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayMemory(max, input_shape, num_actions)
        if continuous:
            self.max_a = 4  # env.action_space.high[0]
            self.min_a = env.action_space.low[0]
            self.continuous = True
        else:
            self.max_a = 4
            self.min_a = 1
            self.continuous = False

        self.actor = Actor(num_actions=num_actions, continuous=continuous)
        self.critic = Critic(continuous=continuous)
        self.target_actor = Actor(num_actions=num_actions, name='target_actor', continuous=continuous)
        self.target_critic = Critic(name='target_critic', continuous=continuous)

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # actor weights
        w = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            w.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(w)
        # critic weights
        w = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            w.append(weight * tau + targets[i] * (1 - tau))
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

    def load_model(self):
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

        if not self.continuous:
            a = a.numpy()

        return a[0]

    def learn(self):
        if self.memory.memory_ctr < self.batch_size:
            return
        a, r, s, new_s, finished = self.memory.sample_buffer(self.batch_size)
        actions = tf.convert_to_tensor(a, dtype=tf.float32)
        states = tf.convert_to_tensor(s, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_s, dtype=tf.float32)
        rewards = tf.convert_to_tensor(r, dtype=tf.float32)

        # UPDATE RULE FOR CRITIC: loss function
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
            if self.continuous:
                print("ACTION ", action)
                print("TESTEST ", np.abs(action[1]) > 0.5)
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
