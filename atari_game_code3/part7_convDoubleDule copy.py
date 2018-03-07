# https://github.com/nilportugues/reinforcement-learning-1/blob/master/Code%202.%20Cartpole/2.%20Double%20DQN/Cartpole_DoubleDQN.py
# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
# duling double

import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse


class double_DQN():
    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.gymEnv = gym.make(environment_name)
        self.stateSize = self.gymEnv.observation_space.shape[0]
        self.isLake = False
        self.actionSize = self.gymEnv.action_space.n
        self.maxSteps = 200 # limit maximum steps for cartpole 
        self.numEpisodes = 1005
        self.learningRate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        # Discount factor : 1: for MountainCar, and 0:99 for CartPole and Space Invaders.
        self.gamma = 0.99
        self.network = "dueling_double"
        self.batch_size = 32
        # create a main model and target model
        self.model = self._createModel()
        self.target_model = self._createModel()
        # initialize the target model so that the parameters in the two models are the same
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _createModel(self):
        model = Sequential()
        states = Input(shape =(self.stateSize,))
        dense1 = Dense(64, activation='relu')(states)

        dense2 = Dense(32, activation='relu')(dense1)
        state_values = Dense(1)(dense2)
        
        dense3 = Dense(32, activation='relu')(dense1)
        advantages = Dense(self.actionSize)(dense3) 


        advantages = Lambda(lambda x: x-tf.reduce_mean(x, axis=1, keep_dims=True))(advantages)
        state_action_values = Add()([state_values, advantages])

        model = Model([states], state_action_values)
        opt = keras.optimizers.Adam(lr=self.learningRate)
        model.compile(optimizer=opt, loss='mse')
        return model

    def replay(self, memory):
        minibatch = memory.sample_batch()

        state_array = []
        target_f_array = []

        # oneHot = np.zeros(self.actionSize)
        # double DQN: 
        # extract information from memory
        # selection of action is from model (what's saved in memory), update is from target model
        for state, action ,reward, next_state, done in minibatch:
            #print("reward from minibatch",reward)
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

        self.model.fit(state_array, target_f_array, batch_size = self.batch_size, epochs = 1, verbose = 0)

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

            while not done:
                steps += 1
                # A_t
                action = self.epsilon_greedy_policy(state)
                # R_t+1, S_t+1
                next_state, reward, done, info = self.gymEnv.step(action)
                
                episode_reward += reward
                # save S_t, A_t, R_t+1, S_t+1 to memory
                next_state = np.reshape(next_state, [1, -1])# shape (1,4)
                
                if done:
                    print("episode", e, "episode_reward", episode_reward)
                    
                    rewards_list.append(episode_reward)
                    next_state = np.zeros(state.shape)
                    memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                    # start new episode
                    state = self.gymEnv.reset()
                    episode_reward = 0
                    # Double DQN
                    # every episode update the target model to be same with model
                    self.update_target_model()

                else:
                    memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                    state = next_state

                # train agent by sampling from memory
                self.replay(memory)


            if (e >= 20 and e%200==0):
                plot_running_mean(rewards_list, "doubleDQN"+self.network)
                # self.test(e)
                # plt.plot(rewards_list)
                # plt.savefig('../result/reward.jpg')


            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                # self.epsilon -= self.epsilon_decay
            

        self.save_model_weights()

    def epsilon_greedy_policy(self, state):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand()<=self.epsilon:
            return self.gymEnv.action_space.sample()

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

    def burn_in_memory(self):
        # udacity, Q_learning_cart.py

        # Initialize your replay memory with a burn_in number of episodes / transitions.
        memory = Replay_Memory()
        state = self.gymEnv.reset()
        if self.isLake==True:
            state = oneHot(state)
        for i in range(memory.burn_in):
            # make a random action
            action = self.gymEnv.action_space.sample()
            next_state, reward, done, info = self.gymEnv.step(action)
            if self.isLake==True:
                next_state = oneHot(next_state)
            if done:
                #if (reward == 1):
                #    print("reached goal")
                # the simulation fails so no next state
                # W * S = 0   = Q(S',A') = 0   if next state is done  
                next_state = np.zeros(state.shape)
                memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                # start new episode
                state = self.gymEnv.reset()
                if self.isLake==True:
                    state = oneHot(state)
                #state, reward, done, info = self.gymEnv.step(env.action_space.sample())
                #if self.isLake==True:
                #    state = oneHot(state)
            else:
                memory.append((np.reshape(state, [1, -1]), action, reward, (np.reshape(next_state, [1, -1])), done))
                state = next_state

        return memory

def main(args):
    # args = parse_arguments()
    # environment_name = 'Stochastic-4x4-FrozenLake-v0'
    # environment_name = 'MountainCar-v0'
    environment_name = 'CartPole-v0'
    
    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)
    #with tf.device('/gpu:0'):
    double_dqn = double_DQN(environment_name)
    train_writer = tf.summary.FileWriter('../log/6doubleDQN', sess.graph)

    print("train double dqn, with replay")
    double_dqn.train()
    train_writer.close()

# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)


