# sac_train.py
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from enviroment import Env_cellular as env

# Set eager off for TF1.x compatibility
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# Hyperparameters
MAX_EPISODES = 400
MAX_EP_STEPS = 100
LR_ACTOR = 0.0002
LR_CRITIC = 0.0004
GAMMA = 0.9
TAU = 0.001
ALPHA = 0.2
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

class ReplayBuffer:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.buffer = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

    def store(self, s, a, r, s_):
        r = np.array([r]) if np.isscalar(r) else np.reshape(r, (1,))
        transition = np.hstack((s.flatten(), a.flatten(), r, s_.flatten()))
        index = self.pointer % MEMORY_CAPACITY
        self.buffer[index, :] = transition
        self.pointer += 1

    def sample(self):
        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.buffer[indices, :]
        s = bt[:, :self.s_dim]
        a = bt[:, self.s_dim:self.s_dim + self.a_dim]
        r = bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim + 1]
        s_ = bt[:, -self.s_dim:]
        return s, a, r, s_

class SAC:
    def __init__(self, s_dim, a_dim, a_bound):
        self.sess = tf.compat.v1.Session()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.buffer = ReplayBuffer(s_dim, a_dim)

        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'state')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'state_')
        self.A = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'action')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'reward')

        with tf.compat.v1.variable_scope('policy'):
            net = tf.compat.v1.keras.layers.Dense(64, activation='relu')(self.S)
            mean = tf.compat.v1.keras.layers.Dense(a_dim)(net)
            log_std = tf.compat.v1.keras.layers.Dense(a_dim, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))(net)

            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)
            normal_sample = tf.random.normal(tf.shape(mean))
            action = tf.tanh(mean + std * normal_sample)
            self.a_sample = action * a_bound

        self.log_prob = -0.5 * ((normal_sample ** 2) + 2 * log_std + np.log(2 * np.pi))
        self.log_prob = tf.reduce_sum(self.log_prob, axis=1, keepdims=True)

        def build_q_network(name, s, a):
            with tf.compat.v1.variable_scope(name):
                inputs = tf.concat([s, a], axis=1)
                net = tf.compat.v1.keras.layers.Dense(64, activation='relu')(inputs)
                net = tf.compat.v1.keras.layers.Dense(64, activation='relu')(net)
                q = tf.compat.v1.keras.layers.Dense(1)(net)
                return q

        self.q1 = build_q_network('q1', self.S, self.A)
        self.q2 = build_q_network('q2', self.S, self.A)
        self.q1_target = build_q_network('q1_target', self.S_, self.a_sample)
        self.q2_target = build_q_network('q2_target', self.S_, self.a_sample)

        self.q1_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
        self.q2_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
        self.q1_target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q1_target')
        self.q2_target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q2_target')

        self.soft_replace = [
            tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
            for t, e in zip(self.q1_target_vars, self.q1_vars)
        ] + [
            tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
            for t, e in zip(self.q2_target_vars, self.q2_vars)
        ]



        q_min = tf.minimum(self.q1_target, self.q2_target)
        self.q_backup = self.R + GAMMA * (q_min - ALPHA * self.log_prob)

        self.q1_loss = tf.reduce_mean(tf.square(self.q_backup - self.q1))
        self.q2_loss = tf.reduce_mean(tf.square(self.q_backup - self.q2))

        self.q1_train = tf.compat.v1.train.AdamOptimizer(LR_CRITIC).minimize(self.q1_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q1'))
        self.q2_train = tf.compat.v1.train.AdamOptimizer(LR_CRITIC).minimize(self.q2_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q2'))

        q_new = build_q_network('q_pi', self.S, self.a_sample)
        self.pi_loss = tf.reduce_mean(ALPHA * self.log_prob - q_new)
        self.pi_train = tf.compat.v1.train.AdamOptimizer(LR_ACTOR).minimize(self.pi_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='policy'))

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a_sample, {self.S: s})[0]

    def store_transition(self, s, a, r, s_):
        self.buffer.store(s, a, r, s_)

    def learn(self):
        if self.buffer.pointer < BATCH_SIZE:
            return
        s, a, r, s_ = self.buffer.sample()
        self.sess.run([self.q1_train, self.q2_train, self.pi_train],
                      {self.S: s, self.A: a, self.R: r, self.S_: s_})
        self.sess.run(self.soft_replace)

# Environment setup
Pn = 1
K = 2
s_dim = 3
a_dim = 1
a_bound = 1
state_am = 1000
location_vector = np.array([[0, 1], [0, 1000]])
location_GF = np.array([[1, 1]])
fading_n = 1
fading_0 = 1

myenv = env(MAX_EP_STEPS, s_dim, location_vector, location_GF, K, Pn, fading_n, fading_0)
sac = SAC(s_dim, a_dim, a_bound)

var = 1
t1 = time.time()
ep_rewardall = []
ep_rewardall_greedy = []
ep_rewardall_random = []
print("Start training")
for i in range(MAX_EPISODES):
    batter_ini = myenv.reset()
    s = myenv.channel_sequence[i % MAX_EP_STEPS, :].tolist()
    s.append(batter_ini)
    s = np.reshape(s, (1, s_dim)) * state_am
    s_greedy = s.copy()
    s_random = s.copy()

    ep_reward = 0
    ep_reward_random = 0
    ep_reward_greedy = 0

    for j in range(MAX_EP_STEPS):
        a = sac.choose_action(s)
        a = np.clip(np.random.normal(a, var), 0, 1)
        r, s_, _ = myenv.step(a, s / state_am, j)
        s_ = s_ * state_am
        sac.store_transition(s, a, r, s_)
        sac.learn()
        s = s_
        ep_reward += r

        r_greedy, s_next_greedy, _ = myenv.step_greedy(s_greedy / state_am, j)
        s_greedy = s_next_greedy * state_am
        ep_reward_greedy += r_greedy

        r_random, s_next_random, _ = myenv.step_random(s_random / state_am, j)
        s_random = s_next_random * state_am
        ep_reward_random += r_random

        if var > 0.1:
            var *= 0.9998

    print('Episode:', i,
          ' Reward: %i' % int(ep_reward),
          ' Reward Greedy: %i' % int(ep_reward_greedy),
          ' Reward Random: %i' % int(ep_reward_random),
          ' Explore: %.2f' % var)

    ep_rewardall.append(ep_reward / MAX_EP_STEPS)
    ep_rewardall_greedy.append(ep_reward_greedy / MAX_EP_STEPS)
    ep_rewardall_random.append(ep_reward_random / MAX_EP_STEPS)

print('Running time: ', time.time() - t1)
plt.plot(ep_rewardall, "^-", label='SAC: rewards')
plt.plot(ep_rewardall_greedy, "+:", label='Greedy: rewards')
plt.plot(ep_rewardall_random, "o--", label='Random: rewards')
plt.xlabel("Episode")
plt.ylabel(" Episodic Reward - Data Rate (NPCU)")
plt.legend()
plt.show()
