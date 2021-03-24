from tensorflow.keras import layers
import numpy as np
import tensorflow as tf


class OUActionNoise:
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

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, critic_model, actor_model, target_actor, target_critic, gamma, state_batch,
            action_batch, reward_batch, next_state_batch, critic_optimizer, actor_optimizer
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, critic_model, actor_model, target_actor, target_critic, gamma,
              critic_optimizer, actor_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(critic_model, actor_model, target_actor, target_critic, gamma, state_batch,
            action_batch, reward_batch, next_state_batch, critic_optimizer, actor_optimizer)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(num_states, num_actions, upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    last_init1 = tf.random_uniform_initializer(minval=-0.002, maxval=0.002)
    last_init2 = tf.random_uniform_initializer(minval=-0.004, maxval=0.004)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(200, activation="relu", kernel_initializer=last_init1)(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init2)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    last_init1 = tf.random_uniform_initializer(minval=-0.002, maxval=0.002)
    last_init2 = tf.random_uniform_initializer(minval=-0.004, maxval=0.004)

    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(400, activation="relu")(concat)
    out = layers.Dense(200, activation="relu", kernel_initializer=last_init1)(out)
    outputs = layers.Dense(1, kernel_initializer=last_init2)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

class DDPG_OFF():
    def __init__(self, num_states, num_actions, lower_bound, upper_bound):
        std_dev = 0.2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        self.buffer = Buffer(num_states, num_actions, 50000, 64)

        self.actor_model = get_actor(num_states, num_actions, upper_bound)
        self.critic_model = get_critic(num_states, num_actions)

        self.target_actor = get_actor(num_states, num_actions, upper_bound)
        self.target_critic = get_critic(num_states, num_actions)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def train(self, env, num_episodes):
        total_episodes = num_episodes
        # Discount factor for future rewards
        gamma = 0.99
        # Used to update target networks
        tau = 0.005

        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        # Takes about 4 min to train
        for ep in range(total_episodes):

            prev_state = env.reset()
            episodic_reward = 0

            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = policy(tf_prev_state, self.ou_noise, self.actor_model,
                                self.lower_bound, self.upper_bound)
                # Recieve state and reward from environment.
                # print("ACTION ", action)
                if env.continuous:
                    action = action[0]
                else:
                    action = int(action[0].argmax())
                # print("ACTION ", action)
                state, reward, done, info = env.step(action)

                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                self.buffer.learn(self.critic_model, self.actor_model, self.target_actor,
                                  self.target_critic, gamma, self.critic_optimizer,
                                  self.actor_optimizer)
                update_target(self.target_actor.variables, self.actor_model.variables, tau)
                update_target(self.target_critic.variables, self.critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
            return avg_reward_list