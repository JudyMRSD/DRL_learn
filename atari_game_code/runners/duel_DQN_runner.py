import os
import sys

import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from keras import backend as K
from keras.models import Model


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers import merge
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Lambda
import random
from numpy.random import seed
from tensorflow import set_random_seed

import numpy as np
import random
import keras
from keras.models import load_model, Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import merge, Input
from keras import backend as K
from collections import deque

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(repo_path)
sys.path.insert(0, os.path.join(repo_path, 'agents'))
sys.path.insert(0, os.path.join(repo_path, 'environment'))
sys.path.insert(0, os.path.join(repo_path, 'utils'))


from environment.gridworld import gameEnv
from agents.duel_DQN import *
from utils.util import *

class duel_DQN_runner():
    def __init__(self):
        self.path = '../model/'
        self.numEpisodes = 1005
        self.learningRate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.model = duelDQN()

    def epsilon_greedy_policy(self, state):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand() <= self.epsilon:
            a = np.random.randint(0, 4)
            return a

        # input shape is [None ,4]
        state = np.reshape(state, [1, -1])
        q_values = self.model.predict(state)
        return np.argmax(q_values)


    def greedy_policy(self, state):
        # Creating greedy policy for test time.
        # input shape is [None ,4]
        state = np.reshape(state, [1, -1])
        q_values = self.model.predict(state)

        return np.argmax(q_values)


    # https://github.com/bstriner/gym-traffic/blob/master/gym_traffic/agents/dqn.py
    # save, load functions
    def save_model_weights(self, suffix=None):
        model_file = '../model/' + self.network
        # Helper function to save your model / weights.
        if suffix == 'weights':
            model_file = model_file + self.network + '_weights.h5'
        else:
            model_file = model_file + self.network + '.h5'

        if not os.path.exists(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        # model.save(filepath)  save as HDF5 file
        self.model.save(model_file)


    # replay()  https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
    def replay(self, memory):
        minibatch = memory.sample_batch()

        # state_array = np.empty((1,self.stateSize,))
        # target_f_array = np.empty((1,self.actionSize))
        state_array = []
        target_f_array = []

        # extract information from memory
        for state, action, reward, next_state, done in minibatch:
            # print("reward from minibatch",reward)
            if done:
                target = reward
            else:
                # discounted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            # Q(S,A)
            target_f = self.model.predict(state)
            # Q(S',A)
            target_f[0][action] = target
            # train the network with state and target_f, default batch size 32
            state_array.append(state)
            target_f_array.append(target_f)

        state_array = np.squeeze(np.array(state_array), axis=1)
        target_f_array = np.squeeze(np.array(target_f_array), axis=1)

        self.model.fit(state_array, target_f_array, batch_size=self.batch_size, epochs=1, verbose=0)


    # training with replay
    def train(self):
        state = self.gymEnv.reset()
        steps = 0

        print("training with replay")
        rewards_list = []
        memory = self.burn_in_memory()
        for e in range(self.numEpisodes):
            # S_t, A_t, R_t+1, S_t+1, A_t+1
            # S_t

            done = False
            episode_reward = 0
            self.epsilon = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(
                -self.decay_rate * steps)
            episode_step = 0
            while not done:
                steps += 1  # total steps during the entire training process
                episode_step += 1  # count of steps inside one episode
                # A_t
                action = self.epsilon_greedy_policy(state)
                # R_t+1, S_t+1
                next_state, reward, done, info = self.gymEnv.step(action)

                episode_reward += reward
                # save S_t, A_t, R_t+1, S_t+1 to memory
                next_state = np.reshape(next_state, [1, -1])  # shape (1,4)

                if done or episode_step > self.maxSteps:
                    print("episode", e, "episode_reward", episode_reward)

                    rewards_list.append(episode_reward)
                    next_state = np.zeros(state.shape)
                    memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                    # start new episode
                    state = self.gymEnv.reset()
                    episode_reward = 0

                else:
                    memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                    state = next_state

                # train agent by sampling from memory
                self.replay(memory)

            if (e >= 20 and e % 100 == 0):
                plot_running_mean(rewards_list, "exponentialDecay_training" + self.network)
                # self.test(e)
                # plt.plot(rewards_list)
                # plt.savefig('../result/reward.jpg')

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                # if self.epsilon > self.epsilon_min:
                # self.epsilon = 1./((e/50) + 10)
                # self.epsilon -= self.epsilon_decay

        self.save_model_weights()

        # return memory


    def burn_in_memory(self):
        # udacity, Q_learning_cart.py

        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory = Replay_Memory()
        state = self.gymEnv.reset()

        for i in range(memory.burn_in):
            # make a random action
            action = self.gymEnv.action_space.sample()
            next_state, reward, done, info = self.gymEnv.step(action)

            if done:
                # if (reward == 1):
                #    print("reached goal")
                # the simulation fails so no next state
                # W * S = 0   = Q(S',A') = 0   if next state is done
                next_state = np.zeros(state.shape)
                memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                # start new episode
                state = self.gymEnv.reset()

            else:
                memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                state = next_state

        return memory



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
