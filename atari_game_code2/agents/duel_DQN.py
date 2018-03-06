# https://yilundu.github.io/2016/12/24/Deep-Q-Learning-on-Space-Invaders.html
# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
# https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
# https://github.com/tokb23/dqn/blob/master/ddqn.py
import os
import sys

import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from keras import backend as K
from keras.models import Model
import tensorflow as tf, numpy as npy, gym, sys, copy, argparse

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.models import Model
from collections import deque

import random
from numpy.random import seed
from tensorflow import set_random_seed
import keras
from keras import backend as K

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

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
from keras.layers import Dense, Input, Lambda, Activation, Add, average, Subtract


repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(repo_path)
sys.path.insert(0, os.path.join(repo_path, 'agents'))
sys.path.insert(0, os.path.join(repo_path, 'environment'))



from gridworld import gameEnv
from duel_DQN import *


class duelDQN():
    def __init__(self, learningRate, actionSize):

        self.actionSize = actionSize

        self.learningRate = learningRate
        self.network = "dueling2"
        self.imgShape = (84, 84, 3)  # num frames = 1, 3 channel image
        self.tau = 0.001

        self.model = self._createModel()
        # duel dqn
        self.target_model = self._createModel()
        # initialize the target model so that the parameters in the two models are the same
        self.update_target_model()

        # sumnmary for loss
        self.loss = tf.placeholder(tf.float32)
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        self.reward = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', tf.reduce_mean(self.reward))

        self.Q = tf.placeholder(tf.float32)
        tf.summary.scalar('Q', tf.reduce_mean(self.Q))
        self.merged = tf.summary.merge_all()

        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # https://yilundu.github.io/2016/12/24/Deep-Q-Learning-on-Space-Invaders.html
    # https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-7-dueling-DQN/
    def _createModel(self):
        h_size = 512
        # model = Sequential()
        input_layer = Input(shape = (84, 84, 3))
        x = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], activation='relu',input_shape=(84, 84, 3))(input_layer)
        x = Conv2D(filters=64, kernel_size=[4,4],strides=[2,2], activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=[3,3],strides=[1,1],activation='relu')(x)
        x = Conv2D(filters=h_size, kernel_size=[7,7],strides=[1,1],activation='relu')(x)
        fc0 = Lambda(lambda a: tf.reshape(a, [-1, h_size]))(x)
        #fc0 = Flatten()(x)
        # fc0 = K.reshape(x, (-1, h_size))

        fc1 = Dense(h_size//2, activation='relu')(fc0)
        state_values = Dense(1)(fc1)
        
        fc2 = Dense(h_size//2, activation='relu')(fc0)
        advantages = Dense(self.actionSize)(fc2) 


        advantages = Lambda(lambda x: x-tf.reduce_mean(x, axis=1, keep_dims=True))(advantages)
        state_action_values = Add()([state_values, advantages])

        model = Model([input_layer], state_action_values)
        opt = keras.optimizers.Adam(lr=self.learningRate)
        model.compile(optimizer=opt, loss='mse')

        return model

    #def combine_A_V(self, x):
    #    return x[0]-K.mean(x[0], axis=1,keepdims= True)+x[1]



class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent.
        # Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        # ramdom.sample(population, k)
        # Return k unique elements chosen from population
        sample_batch = random.sample(self.memory, batch_size)
        return sample_batch

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)



