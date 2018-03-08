# https://github.com/nilportugues/reinforcement-learning-1/blob/master/Code%202.%20Cartpole/2.%20Double%20DQN/Cartpole_DoubleDQN.py
# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
# duling double

import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from gridworld import gameEnv


import matplotlib.pyplot as plt
import os
import sys

import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from keras import backend as K
from keras.models import Model


from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers import merge
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Lambda
import random
from numpy.random import seed
from tensorflow import set_random_seed

from __future__ import division

import numpy as np
import random
import keras
from keras.models import load_model, Sequential, Model
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import merge, Input
from keras import backend as K
from collections import deque
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Input, Lambda, Activation, Add, average, Subtract


from keras.callbacks import TensorBoard


from keras.callbacks import History

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from helper import *
from keras import backend as K

# ### Load the game environment

# Feel free to adjust the size of the gridworld. Making it smaller provides an easier task for our DQN agent, while making the world larger increases the challenge.

# In[2]:


from gridworld import gameEnv

env = gameEnv(partial=False, size=5)


# Above is an example of a starting environment in our simple game. The agent controls the blue square, and can move up, down, left, or right. The goal is to move to the green square (for +1 reward) and avoid the red square (for -1 reward). The position of the three blocks is randomized every episode.

# ### Implementing the network itself

# In[3]:


class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                 padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1],
                                 padding='VALID', biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # print("Qout",tf.shape(self.Qout))# Q(s,a1), Q(s,a2), shape=(2,)
        self.predict = tf.argmax(self.Qout, 1)  # chosen action
        print("predict", tf.shape(self.predict))  # shape=(1,)

        self.maxQ = tf.reduce_max(tf.reduce_max(self.Qout, axis=1))
        print("maxQ", tf.shape(self.maxQ))  # shape=(1,)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)  # shape=(1,)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)  # shape=(1,)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)  # shape=(2,)
        # print("targetQ, actions, onehot",tf.shape(self.targetQ), tf.shape(self.actions), tf.shape(self.actions_onehot ))
        # "Shape:0", shape=(1,), dtype=int32) Tensor("Shape_1:0", shape=(1,), dtype=int32) Tensor("Shape_2:0", shape=(2,), dtype=int3
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        # self.meanQ = tf.reduce_mean(self.Q)
        # print("Q",tf.shape(self.Q)) # shape=(1,)    Q(s, a_i), a_i is the action encoded by actions_onehot, e.g. 01 is action a1, 10 is action a2
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        # predicted q
        self.q_s_a = tf.placeholder(tf.float32)
        tf.summary.scalar('q_s_a', self.q_s_a)
        # greedy prediction
        tf.summary.scalar('maxQ', self.maxQ)

        # loss
        tf.summary.scalar('loss', self.loss)
        # reward
        self.reward = tf.placeholder(tf.int32)
        tf.summary.scalar('reward', self.reward)
        self.merged = tf.summary.merge_all()


# ### Experience Replay

# This class allows us to store experies and sample then randomly to train the network.

# In[4]:


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# This is a simple function to resize our game frames.

# In[5]:


def processState(states):
    return np.reshape(states, [21168])


# These functions allow us to update the parameters of our target network with those of the primary network.

# In[6]:


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# ### Training the network

# Setting all the training parameters

# In[7]:


batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 10000  # How many steps of random actions before training begins.
max_epLength = 50  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "./dqn"  # The path to save our model to.
h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network

print("hello")

# In[ ]:


tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()
'''
for i in range(len(trainables)):
    print("i=",i," variable=",trainables[i].name)
i= 0  variable= Conv/weights:0
i= 1  variable= Conv_1/weights:0
i= 2  variable= Conv_2/weights:0
i= 3  variable= Conv_3/weights:0
i= 4  variable= Variable:0
i= 5  variable= Variable_1:0
i= 6  variable= Conv_4/weights:0
i= 7  variable= Conv_5/weights:0
i= 8  variable= Conv_6/weights:0
i= 9  variable= Conv_7/weights:0
i= 10  variable= Variable_2:0
i= 11  variable= Variable_3:0
'''
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    episodeBuffer = experience_buffer()

    train_writer = tf.summary.FileWriter('./log', sess.graph)

    for i in range(num_episodes):

        # Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # The Q-Network

        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.

            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            q_s_a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

            rAll += r

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                # skip frame to accelerate the training process
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    # trainBatch[:,3]: s'
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    # for i in range (10):
                    #    plt.imshow(np.reshape(trainBatch[i,3], [84,84,3]))
                    #    plt.show()


                    # end_multiplier = 0 if it's the end state (done = 1)
                    # else,trainBatch[:,4]=0, -(0-1)=1 ,  end_multiplier = 1
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # s,a,r,s1,d
                    # actions = trainBatch[:,1]
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    # episodeReward = sess.run(mainQN.reward, feed_dict={mainQN.reward: rAll})
                    summary = sess.run(mainQN.merged,
                                       feed_dict={mainQN.reward: rAll, mainQN.q_s_a: q_s_a, mainQN.targetQ: targetQ,
                                                  mainQN.actions: trainBatch[:, 1],
                                                  mainQN.scalarInput: np.vstack(trainBatch[:, 0])})

                    train_writer.add_summary(summary, total_steps)
                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.

            s = s1

            if d == True:
                print("j=", j)
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        # summary = sess.run(self.mainQN.merged, feed_dict={duelDQN.reward: rAll})
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if len(rList) % 20 == 0:
            print("total_steps", total_steps, "mean reward", np.mean(rList[-10:]), "epsilon", e)
            # plot_running_mean(rList, "double-dueling")
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')

    train_writer.close()

print("END of training, Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

# ### Checking network learning

# Mean reward over time

# In[ ]:


rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)

