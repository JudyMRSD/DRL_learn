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


from gridworld import gameEnv
from duel_DQN import *
from util import *
from keras.callbacks import TensorBoard


from keras.callbacks import History 

class duel_DQN_runner():
    def __init__(self):
        self.path = '../model/'
        self.network = 'duel_DQN_runner'
        self.numEpisodes = 10000
        self.learningRate = 0.0001
        self.epsilon_start = 1
        self.epsilon = self.epsilon_start
        self.epsilon_end = 0.1
        self.annealingSteps = 10000
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end)/self.annealingSteps
        self.gamma = 0.99
        self.gymEnv = gameEnv(partial=False, size=5)
        self.actionSize = self.gymEnv.actions
        self.duelDQN = duelDQN(self.learningRate, self.actionSize)
        self.model = self.duelDQN.model
        self.target_model = self.duelDQN.target_model
        self.batch_size = 32
        self.skip = 4
        self.max_epLength = 50
        self.update_Q_steps = 1000

    def epsilon_greedy_policy(self, state):

        state = np.reshape(state, [-1, 84,84,3])
        # predict use self.model, not target.model
        q_values = self.model.predict(state)

        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand(1) <= self.epsilon:
            a = np.random.randint(0, 4)
            return a, q_values
        else:
            return np.argmax(q_values), q_values


    def greedy_policy(self, state):
        # Creating greedy policy for test time.
        # input shape is [None ,4]
        state = np.reshape(state, [-1, 84, 84, 3])
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
        history = History()

        minibatch = memory.sample_batch()

        # state_array = np.empty((1,self.stateSize,))
        # target_f_array = np.empty((1,self.actionSize))
        state_array = []
        target_f_array = []

        # extract information from memory
        for memory in minibatch:

            state, action, reward, next_state, done = memory[0][0], memory[0][1], memory[0][2], memory[0][3], memory[0][4]
            # print("state, action, reward, next_state, done", state, action, reward, next_state, done)
            state = np.reshape(state, [-1,84,84,3])
            next_state = np.reshape(next_state, [-1,84, 84, 3])

            if done:
                target = reward
            else:
                # discounted reward
                # like Q learning, get maximum Q value at S' but from target model
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state))

            # Q(S,A)
            target_f = self.model.predict(state)
            # Q(S',A)
            target_f[0][action] = target
            # train the network with state and target_f, default batch size 32
            state_array.append(state)
            target_f_array.append(target_f)

        state_array = np.squeeze(np.array(state_array), axis=1)
        target_f_array = np.squeeze(np.array(target_f_array), axis=1)
        # print("state array", state_array.shape, "   target f array",target_f_array.shape)# state array (32, 84, 84, 3)    target f array (32, 4)
        self.model.fit(state_array, target_f_array, batch_size=self.batch_size, epochs=1, verbose=0, callbacks=[history])
        print("loss",history.history['loss'])
        loss =history.history['loss']
        return loss

    # training with replay
    def train(self,sess):
        test_writer = tf.summary.FileWriter('../summary/DQN_dueling', sess.graph)

        state = self.gymEnv.reset()
        state = self.processState(state)
        total_steps = 0
        print("training with replay")
        rewards_list = []
        memory = self.burn_in_memory()

        for e in range(self.numEpisodes):
            
            # S_t, A_t, R_t+1, S_t+1, A_t+1
            # S_t
            # print("e", e)
            done = False
            episode_reward = 0
            _last_reward = 0
            ep_length = 0 
            while not done and ep_length<self.max_epLength:
                ep_length += 1
                total_steps += 1

                if self.epsilon > self.epsilon_end:
                    self.epsilon -= self.epsilon_decay


                # A_t
                action, q_values = self.epsilon_greedy_policy(state)
                # R_t+1, S_t+1
                next_state, reward, done = self.gymEnv.step(action)
                next_state = self.processState(next_state)
                episode_reward += reward

                # save S_t, A_t, R_t+1, S_t+1 to memory


                if done or ep_length>=self.max_epLength:

                    rewards_list.append(episode_reward)
                    _last_reward = episode_reward
                    next_state = np.zeros(state.shape)
                    next_state = self.processState(next_state)

                    memory.append(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))

                    # start new episode
                    state = self.gymEnv.reset()
                    episode_reward = 0

                else:
                    memory.append(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))
                    state = next_state

                # train agent by sampling from memory
                # skip frame to speed up the process
                if total_steps % self.skip == 0:
                    loss = self.replay(memory)
                    # tf.summary.scalar("loss", loss)
                    # self.duelDQN.update_target_model()
                    summary = sess.run(self.duelDQN.merged, feed_dict={self.duelDQN.loss:loss, self.duelDQN.Q:np.max(q_values),
                        self.duelDQN.reward: _last_reward})
                    test_writer.add_summary(summary, total_steps)

                if total_steps % self.update_Q_steps == 0:
                # update target network
                    self.duelDQN.update_target_model()

                
            if (e >= 10 and e % 100 == 0):
                mean_reward = np.mean(rewards_list[-10:])
                print("\n","episode=", e, " total_steps=",total_steps," mean reward=",mean_reward, " epsilon=", self.epsilon)
                # tf.summary("mean_reward", mean_reward)
                #if (e%20==0):
                plot_running_mean(rewards_list, "duelDQN" + self.network)
                #print("losslis", loss_list)
                #plotLoss(loss_list, "loss" + self.network)
            
        self.save_model_weights()

        # return memory

    def processState(self, state):
        return np.reshape(state, [84*84*3])  # 84x84x3 channel image reshaped to 1d vector

    def burn_in_memory(self):
        # udacity, Q_learning_cart.py

        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory = Replay_Memory()
        state = self.gymEnv.reset()
        state = self.processState(state)
        for i in range(memory.burn_in):
            # make a random action
            action = np.random.randint(0,4)
            next_state, reward, done = self.gymEnv.step(action)
            next_state = self.processState(next_state)
            if done:
                # if (reward == 1):
                #    print("reached goal")
                # the simulation fails so no next state
                # W * S = 0   = Q(S',A') = 0   if next state is done
                next_state = np.zeros(state.shape)
                memory.append(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))
                # start new episode
                state = self.gymEnv.reset()
                state = self.processState(state)

            else:
                memory.append(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))
                state = next_state

        return memory





