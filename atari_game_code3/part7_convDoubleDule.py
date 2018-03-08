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
from keras.utils.vis_utils import plot_model


from keras.callbacks import TensorBoard


from keras.callbacks import History 



class duelDQN():
    def __init__(self):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.gymEnv = gameEnv(partial=False, size=5)
        #self.stateSize = self.gymEnv.observation_space.shape[0]
        self.actionSize = self.gymEnv.actions
        self.maxSteps = 50 # limit maximum steps for cartpole
        self.numEpisodes = 10000
        self.learningRate = 0.0001
        

        self.epsilon_start = 1
        self.epsilon = self.epsilon_start
        self.epsilon_end = 0.1

        self.annealingSteps = 10000
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end)/self.annealingSteps


        self.max_epLength = 50

        # Discount factor : 1: for MountainCar, and 0:99 for CartPole and Space Invaders.
        self.gamma = 0.99
        self.network = "dueling_double"
        self.batch_size = 32


        self.skip = 4


        # create a main model and target model
        self.model = self._createModel()
        self.target_model = self._createModel()
        # initialize the target model so that the parameters in the two models are the same
        self.update_target_model()


        # sumnmary for loss
        self.loss = tf.placeholder(tf.float32)
        # print("loss shape",tf.shape(self.loss)) #shape=(?,)
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        self.reward = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', self.reward)

        self.currentEpsilon = tf.placeholder(tf.float32)
        tf.summary.scalar('epsilon', self.currentEpsilon)

        self.Q = tf.placeholder(tf.float32)
        tf.summary.scalar('Q', self.Q)
        self.merged = tf.summary.merge_all()



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _createModel(self):
        h_size = 512
        # model = Sequential()
        input_layer = Input(shape = (84, 84, 3))
        x = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], activation='relu',input_shape=(84, 84, 3))(input_layer)
        x = Conv2D(filters=64, kernel_size=[4,4],strides=[2,2], activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=[3,3],strides=[1,1],activation='relu')(x)
        x = Conv2D(filters=h_size, kernel_size=[7,7],strides=[1,1],activation='relu')(x)
        
        x_value = Lambda(lambda x: x[:,:,:,:h_size//2])(x)
        
        x_advantage = Lambda(lambda x: x[:,:,:,h_size//2:])(x)


        x_value = Lambda(lambda a: tf.reshape(a, [-1, h_size//2]))(x_value)

        x_advantage = Lambda(lambda a: tf.reshape(a, [-1, h_size//2]))(x_advantage)

        #Process spliced data stream into value and advantage function
        state_values = Dense(1, activation="linear")(x_value)
        advantages = Dense(self.actionSize, activation="linear")(x_advantage)

        advantages = Lambda(lambda x: x-tf.reduce_mean(x, axis=1, keep_dims=True))(advantages)
        state_action_values = Add()([state_values, advantages])

        model = Model([input_layer], state_action_values)
        opt = keras.optimizers.Adam(lr=self.learningRate)
        model.compile(optimizer=opt, loss='mse')

        plot_model(model, to_file='./modelPlot/duel.png',show_shapes=True)

        return model

    def replay(self, memory):
        history = History()

        minibatch = memory.sample_batch()

        state_array = []
        target_f_array = []

        # oneHot = np.zeros(self.actionSize)
        # double DQN: 
        # extract information from memory
        # selection of action is from model (what's saved in memory), update is from target model
        for state, action ,reward, next_state, done in minibatch:
            #print("reward from minibatch",reward)

            #print("state shape", state.shape)
            #print("next stat eshape", next_state.shape)

            if done:
                target = reward
            else:
                # double DQN: 
                # discounted reward
                # like Q learning, get maximum Q value at S' but from target model
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state))

            # Q(S,A) from model
            target_f = self.model.predict(state)
            # Q(S',A)
            target_f[0][action] = target
            # train the network with state and target_f, default batch size 32
            state_array.append(state)
            target_f_array.append(target_f)
            
        state_array = np.squeeze(np.array(state_array), axis=1)
        target_f_array = np.squeeze(np.array(target_f_array), axis =1 )
        #for i in range(10):
        #    plt.imshow(np.reshape(state_array[i], [84, 84, 3]))
        #    plt.title(i)
        #    plt.show()

        self.model.fit(state_array, target_f_array, batch_size = self.batch_size, epochs = 1, verbose = 0 , callbacks=[history])
        # print("loss",history.history['loss'])
        loss =history.history['loss']
        return loss

    def train(self, sess):
        train_writer = tf.summary.FileWriter('./summary/', sess.graph)
        rList = []
        state = self.gymEnv.reset()
        print("state shape", state.shape)
        total_steps = 0

        print("training with replay")
        rewards_list = []
        memory = self.burn_in_memory()
        for e in range(self.numEpisodes):
            # S_t, A_t, R_t+1, S_t+1, A_t+1
            # S_t
            
            done = False
            episode_reward = 0

            _last_reward = 0
            ep_length = 0 


            while not done and ep_length<self.max_epLength:
                ep_length += 1
                total_steps += 1

                # A_t
                action, q_values = self.epsilon_greedy_policy(state)
                # R_t+1, S_t+1
                next_state, reward, done = self.gymEnv.step(action)


                state = np.reshape(state, [-1, 84,84,3])
                next_state = np.reshape(next_state, [-1, 84,84,3])

                #print("next state shape", next_state.shape)


                episode_reward += reward
                #  save S_t, A_t, R_t+1, S_t+1 to memory
                if done or ep_length>=self.max_epLength:
                    # print("episode", e, "episode_reward", episode_reward, "total steps", total_steps, "epsilon", self.epsilon)
                    
                    rewards_list.append(episode_reward)
                    _last_reward = episode_reward
                    if done:
                        next_state = np.zeros(state.shape)
                    #print("next state shape", next_state.shape)
                    # s,a,r,s1,d
                    #plt.imshow(state)
                    #plt.imshow(next_state)
                    #plt.show()

                    state = np.reshape(state, [-1, 84,84,3])
                    next_state = np.reshape(next_state, [-1, 84,84,3])


                    memory.append((state, action, reward, next_state, done))
                    # start new episode
                    state = self.gymEnv.reset()
                    state = np.reshape(state, [-1, 84,84,3])

                    episode_reward = 0
                    # Double DQN
                    # every episode update the target model to be same with model
                    # question: how often update model?
                    self.update_target_model()

                else:

                    #plt.imshow(state)
                    #plt.imshow(next_state)
                    #plt.show()


                    state = np.reshape(state, [-1, 84,84,3])
                    next_state = np.reshape(next_state, [-1, 84,84,3])

                    memory.append((state, action, reward, next_state, done))
                    state = next_state
                if total_steps % self.skip == 0:
                    # train agent by sampling from memory
                    loss = self.replay(memory)

                    #summary = sess.run(self.merged, feed_dict={self.loss:loss, self.Q:np.max(q_values),
                    #    self.reward: _last_reward,  self.currentEpsilon:self.epsilon})
                    summary = sess.run(self.merged, feed_dict={self.loss:loss, self.Q:np.max(q_values),
                        self.reward: episode_reward,  self.currentEpsilon:self.epsilon})
                    train_writer.add_summary(summary, total_steps)


                if self.epsilon > self.epsilon_end:
                        self.epsilon -= self.epsilon_decay
            rList.append(episode_reward)
            if len(rList) % 20 == 0:
                    print("total_steps", total_steps, "mean reward", np.mean(rList[-10:]), "epsilon", self.epsilon)
            


    def epsilon_greedy_policy(self, state):



        state = np.reshape(state, [-1, 84,84,3])
        # predict use self.model, not target.model
        q_values = self.model.predict(state)

        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand()<=self.epsilon:
            a = np.random.randint(0, 4)
            return a, q_values

        else:
            return np.argmax(q_values), q_values
        

    def greedy_policy(self, state):
        state = np.reshape(state, [-1, 84, 84, 3])
        q_values = self.model.predict(state)

        return np.argmax(q_values)


    def burn_in_memory(self):
        # udacity, Q_learning_cart.py

        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory = Replay_Memory()
        state = self.gymEnv.reset() # question: should I reshape it before storing?
        #print("state shape after reset", state.shape)
        #plt.imshow(state)
        #plt.show()

        for i in range(memory.burn_in):
            # make a random action
            action = np.random.randint(0,4)

            next_state, reward, done = self.gymEnv.step(action)

            #plt.imshow(next_state)
            #plt.show()


            if done:
                next_state = np.zeros(state.shape)
                #print("next state shape", next_state.shape, "reward=", reward)
                # s,a,r,s1,d
                
                state = np.reshape(state, [-1, 84,84,3])
                next_state = np.reshape(next_state, [-1, 84,84,3])



                memory.append((state, action, reward, next_state, done))

                # start new episode
                state = self.gymEnv.reset()
                episode_reward = 0

            else:
                # print("state shape after else", state.shape, "reward=", reward)
                state = np.reshape(state, [84,84,3])
                # plt.imshow(state)

                state = np.reshape(state, [-1, 84,84,3])
                next_state = np.reshape(next_state, [-1, 84,84,3])


                memory.append((state, action, reward, next_state, done))
                state = next_state

        return memory


class Replay_Memory():
    def __init__(self, memory_size=2000, burn_in=1000):
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


def main(args):
    # args = parse_arguments()
    # environment_name = 'Stochastic-4x4-FrozenLake-v0'
    # environment_name = 'MountainCar-v0'
    
    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)
    #with tf.device('/gpu:0'):
    duel = duelDQN()

    print("train double dqn, with replay")
    duel.train(sess)


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)


