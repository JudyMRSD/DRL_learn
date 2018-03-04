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
        self.tau = 1000

        self.model = self._createModel()
        # duel dqn
        self.target_model = self._createModel()
        # initialize the target model so that the parameters in the two models are the same
        self.update_target_model()
        

    def update_target_model(self):
        # print("weights[0]",self.model.get_weights()[0]) # [[[[3.62457298e-02 ...
        # print("weights[0]*100",self.model.get_weights()[0]*100)# [[[[3.62457299e+00  ...
        
        # targetweight = modelweight * tau + targetweight*(1-tau)
        # op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        # https://github.com/awjuliani/DeepRL-Agents/blob/16ebef4bc36689c2d1859543b8c23aec8cda3a99/helper.py
        old_W = self.model.get_weights()
        oldTarget_W = self.target_model.get_weights()
        tauVec = [self.tau]* len(old_W)
        newTarget_W = np.multiply(old_W,tauVec) + np.multiply(oldTarget_W,tauVec)
        # newTarget_W = old_W*tauVec + oldTarget_W*tauVec
        self.target_model.set_weights(newTarget_W)


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



