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
        self.network = "dueling"
        self.imgShape = (84, 84, 3)  # num frames = 1, 3 channel image

        self.model = self._createModel()
        # duel dqn
        self.target_model = self._createModel()
        # initialize the target model so that the parameters in the two models are the same
        self.update_target_model()

    def update_target_model(self):
        print("self.model.get_weights()",len(self.model.get_weights()))# 12
        print("weights", self.model.get_weights())
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
        x = Flatten()(x)
        x_value = Lambda(lambda x: x[:,:h_size//2])(x)
        x_advantage = Lambda(lambda x: x[:,h_size//2:])(x)

        #Process spliced data stream into value and advantage function
        value = Dense(1, activation="linear")(x_value)
        advantage = Dense(self.actionSize, activation="linear")(x_advantage)

        prediction = Lambda(self.combine_A_V, output_shape =(self.actionSize,))([advantage, value])
        model = Model(input = [input_layer], output=[prediction])

        # plot model 
        plot_model(model, to_file='../result/duelingDQN_model_exponential.png',show_shapes=True)

        # mean squared loss  = (Q_target - Q) ^2
        opti = Adam(lr=self.learningRate)
        model.compile(loss='mse', optimizer=opti)

        '''
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()
        
        for name, weight in zip(names, weights):
            print(name, weight.shape)
        conv2d_1/kernel:0 (8, 8, 3, 32)
        conv2d_1/bias:0 (32,)
        conv2d_2/kernel:0 (4, 4, 32, 64)
        conv2d_2/bias:0 (64,)
        conv2d_3/kernel:0 (3, 3, 64, 64)
        conv2d_3/bias:0 (64,)
        conv2d_4/kernel:0 (7, 7, 64, 512)
        conv2d_4/bias:0 (512,)
        dense_2/kernel:0 (256, 4)
        dense_2/bias:0 (4,)
        dense_1/kernel:0 (256, 1)
        dense_1/bias:0 (1,)
        # target
        conv2d_5/kernel:0 (8, 8, 3, 32)
        conv2d_5/bias:0 (32,)
        conv2d_6/kernel:0 (4, 4, 32, 64)
        conv2d_6/bias:0 (64,)
        conv2d_7/kernel:0 (3, 3, 64, 64)
        conv2d_7/bias:0 (64,)
        conv2d_8/kernel:0 (7, 7, 64, 512)
        conv2d_8/bias:0 (512,)
        dense_4/kernel:0 (256, 4)
        dense_4/bias:0 (4,)
        dense_3/kernel:0 (256, 1)
        dense_3/bias:0 (1,)

        '''
        return model

    def combine_A_V(self, x):
        return x[0]-K.mean(x[0])+x[1]



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



