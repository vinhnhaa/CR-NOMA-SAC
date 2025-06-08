# SAC version replacing DDPG
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from enviroment import Env_cellular as env

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#####################  hyper parameters  ####################
ct = 200
MAX_EPISODES = 20
MAX_EP_STEPS = 100
LR_ACTOR = 0.0002
LR_CRITIC = 0.0004
GAMMA = 0.9
TAU = 0.001
ALPHA = 0.2
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

# ========================= SAC Modules ========================
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
            net = tf.compat.v1.keras.layers.Dense(64, activation='relu')(net)
            mean = tf.compat.v1.keras.layers.Dense(a_dim)(net)
            log_std = tf.compat.v1.keras.layers.Dense(a_dim)(net)
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

        self.q1_train = tf.compat.v1.train.AdamOptimizer(LR_CRITIC).minimize(
            self.q1_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q1'))
        self.q2_train = tf.compat.v1.train.AdamOptimizer(LR_CRITIC).minimize(
            self.q2_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='q2'))

        q_new = build_q_network('q_pi', self.S, self.a_sample)
        self.pi_loss = tf.reduce_mean(ALPHA * self.log_prob - q_new)
        self.pi_train = tf.compat.v1.train.AdamOptimizer(LR_ACTOR).minimize(
            self.pi_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='policy'))

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
# Training loop with K variations
s_dim = 3
a_dim = 1
a_bound = 1
state_am = 10000
K_range = np.array([2, 4, 6, 8, 10])
rate_SAC = []
rate1_SAC = []
rate10_SAC = []
rate_greedy = []
rate_random = []

for k in range(len(K_range)):
    K = K_range[k]
    ratect_SAC = 0
    ratect1 = 0
    ratect10 = 0
    ratect_greedy = 0
    ratect_random = 0

    for mct in range(ct):
        var = 1
        sac = SAC(s_dim, a_dim, a_bound)

        locationspace = np.linspace(1, 1000, num=K)
        location_vector = np.zeros((K, 2))
        location_vector[:, 1] = locationspace
        location_GF = np.array([[1, 1]])

        hnx1 = np.random.randn(K, 2)
        hnx2 = np.random.randn(K, 2)
        fading_n = hnx1 ** 2 + hnx2 ** 2
        h0x1 = np.random.randn(1, 1)
        h0x2 = np.random.randn(1, 1)
        fading_0 = h0x1[0, 0] ** 2 + h0x2[0, 0] ** 2

        Pn = 10 ** ((30 - 30) / 10)
        myenv = env(MAX_EP_STEPS, s_dim, location_vector, location_GF, K, Pn, fading_n, fading_0)
        rate10 = 0
        ratek_SAC = 0
        ratek_greedy = 0
        ratek_random = 0

        for i in range(MAX_EPISODES):
            batter_ini = myenv.reset()
            s = myenv.channel_sequence[i % myenv.K, :].tolist()
            s.append(batter_ini)
            s = np.reshape(s, (1, s_dim)) * state_am
            s_greedy = s.copy()
            s_random = s.copy()

            reward_sac_vector = []
            reward_greedy_vector = []
            reward_random_vector = []

            for j in range(MAX_EP_STEPS):
                a = sac.choose_action(s)
                a = np.clip(np.random.normal(a, var), 0, 1)
                r, s_, _ = myenv.step(a, s / state_am, j)
                s_ = s_ * state_am
                sac.store_transition(s, a, r, s_)
                sac.learn()
                s = s_
                reward_sac_vector.append(r)

                r_greedy, s_next_greedy, _ = myenv.step_greedy(s_greedy / state_am, j)
                s_greedy = s_next_greedy * state_am
                reward_greedy_vector.append(r_greedy)

                r_random, s_next_random, _ = myenv.step_random(s_random / state_am, j)
                s_random = s_next_random * state_am
                reward_random_vector.append(r_random)

                if var > 0.05:
                    var *= .9998

            ratek_SAC = sum(reward_sac_vector) / MAX_EP_STEPS
            if i == 0:
                rate1 = ratek_SAC
            if i == 9:
                rate10 = ratek_SAC

            ratek_greedy = sum(reward_greedy_vector) / MAX_EP_STEPS
            ratek_random = sum(reward_random_vector) / MAX_EP_STEPS
            print(ratek_SAC)
            print(f"Iteration., {k}--{mct},  Episode:, {i},  Reward: {ratect_SAC/ct },")
            # print(f"[K={K}] mct={mct}, episode={i}, SAC_reward={ratek_SAC:.2f}, greedy={ratek_greedy:.2f}, random={ratek_random:.2f}")

        tf.keras.backend.clear_session()
        ratect1 += rate1
        ratect10 += rate10
        ratect_SAC += ratek_SAC
        ratect_greedy += ratek_greedy
        ratect_random += ratek_random

    rate_SAC.append(ratect_SAC / ct)
    rate1_SAC.append(ratect1 / ct)
    rate10_SAC.append(ratect10 / ct)
    rate_greedy.append(ratect_greedy / ct)
    rate_random.append(ratect_random / ct)

print(f"rate_greedy is {rate_greedy} and rate_random is {rate_random}")
K_rangex = ['2', '4', '6', '8', '10']
plt.plot(K_rangex, rate1_SAC, "^--", label='SAC (1 episode)')
plt.plot(K_rangex, rate10_SAC, "^-", label='SAC (10 episodes)')
plt.plot(K_rangex, rate_SAC, "^-", label='SAC (20 episodes)')
plt.plot(K_rangex, rate_greedy, "+:", label='Greedy')
plt.plot(K_rangex, rate_random, "o--", label='Random')
plt.xlabel("The Number of Primary Users (K)")
plt.ylabel("Average Data Rate (NPCU)")
plt.legend(loc='best')
plt.show()
