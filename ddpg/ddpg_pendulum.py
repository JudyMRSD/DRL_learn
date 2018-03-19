import random
import numpy as np
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K
import sys
import gym

# reference: Udacity deep learning tutorial
# https://classroom.udacity.com/nanodegrees/nd101/parts/7d0218b1-1a81-4d49-95f7-14b015020851/modules/691b7845-f7d8-413d-90c7-971cd5016b5c/lessons/fef7e79a-0941-460b-936c-d24c759ff700/concepts/d254347a-68f4-47d0-912a-33fd79719cf8

class ReplayBuffer:
    def __init__(self, memory_size=2000, burn_in=1000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent.
        # Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        # ramdom.sample(population, k)
        # Return k unique elements chosen from population
        sample_batch = random.sample(self.memory, batch_size)
        return sample_batch

    def append(self, state, action, reward, next_state, done):
        # Appends transition to the memory.
        # state, action, reward, next_state, done
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high-self.action_low
        # build actor model
        self.build_model()

    def build_model(self):
        '''
        input : state
        :return:
        action under a deterministic policy
        '''
        states = layers.Input(shape=(self.state_size,), name = 'states')
        # add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)
        # Add final output layer with sigmoid activation
        # especially used for models where we have to predict the probability as an output.
        # Since probability of anything exists only between the range of 0 and 1
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        # scale [0,1] output for each action dimension to proper range [action_low, action_high]
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_low, name='actions')(raw_actions)
        # create model
        self.model = models.Model(inputs=states, outputs=actions)
        # Define loss function using action value (Q value) gradients
        # the gradient will be provided by the critic network
        action_gradients = layers.Input(shape=(self.action_size,))
        # question: how was the loss defined?
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        # inside get_updates:
        # 1. get_gradients(loss, params)     dL/dW
        # 2. question : what does get_updates do?
        # https://github.com/keras-team/keras/blob/master/keras/optimizers.py
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss = loss)
        # keras.backend.function(inputs, outputs, updates = None)
        # inputs: List of placeholder tensors
        # outputs: List of output tensors
        # updates: List of update ops
        # K.learning_phase: The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed
        # as input to any Keras function that uses a different behavior at train time and test time.

        # input: similar to tensorflow feed_dict
        # updates: updates_op.run(feed_dict=....
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs = [],
            updates = updates_op
        )

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        '''
        input: (state, action) pairs
        :return: Q-values
        '''
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name = 'states')
        actions = layers.Input(shape=(self.action_size,), name ='actinos')
        # hidden layer for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        # hidden layer for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        # Hyper parameters: layer size, activations, batch normalization, regularization

        # combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Final output layer to produce action values(Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with builtin loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # compute action gradients (derivative of Q values w.r.t actions)
        action_gradients = K.gradients(Q_values, actions)

        # define an additional function to fetch action gradients (for actor model)
        # a separate function needs to be defined to provide access to these gradients:
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients
        )

class DDPG():
    def __init__(self, task, env):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.env = env
        self.numEpisodes = 10000
        # two copies of each model - one local and one target
        # This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning,
        # and is used to decouple the parameters being updated from the ones that are producing target values.

        # Actor (Policy) Model:
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model:
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        # size, mu, theta, sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay Memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.burn_in = 1000
        self.memory = ReplayBuffer(self.buffer_size, self.burn_in)

        # Hyper parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.01 # soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        # state = self.task.reset()
        # S_t
        # state = self.env.reset()
        state = self.env.reset()
        return state

    def step(self, state, action, episodeReward):
        next_state, reward, done, info = self.env.step(action)
        # Save experience / reward
        self.memory.append(state, action, reward, next_state, done)
        # learn, if enough samples are available in memory
        #if len(self.memory) > self.batch_size:
        experiences = self.memory.sample_batch()
        self.learn(experiences)
        # Roll over last state and action
        state = next_state
        return state, episodeReward

    def act(self, states):
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample()) # add some noise for exploration

    def learn(self, experiences):
        # update policy and value parameters using given batch of experience tuples
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc)
        # question: vstack vs np.array ?
        # question: action shape?
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # question: why use next state to predict actions_next ?
        # use next state to predict Q is reasonable (TD target)
        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # compute Q targets for current states and train critic model(local)
        # self.gamma * Q_targets_next = 0 if done=True
        # TD target
        Q_targets = rewards + self.gamma * Q_targets_next * (1-dones)
        # update critic by minimizing the loss : L = mean square error (y_i=Q_targets , Q(s,a)=feed forward using (states, actions) pair)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        # Train actor model (local)
        # question: what is action size ?
        # action_gradients = dQ / da
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        # custom training function
        # inputs=[self.model.input, action_gradients, K.learning_phase()]
        # outputs = [],   updates = updates_op
        self.actor_local.train_fn([states, action_gradients, 1])
        # soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        new_weights = self.tau*local_weights+ (1-self.tau)*target_weights
        target_model.set_weights(new_weights)


    def burn_in_memory(self):
        self.memory = ReplayBuffer()
        state = self.env.reset() # question: should I reshape it before storing?
        for i in range(self.memory.burn_in):
            # make a random action
            action = self.env.action_space.sample()

            next_state, reward, done, info = self.env.step(action)

            self.memory.append(state, action, reward, next_state, done)

            if done:
                state = self.env.reset()


    def train(self):
        # reset self.noise and self.last_state
        state = self.reset_episode();
        total_steps = 0

        print("training with replay")
        rList = []
        self.burn_in_memory()
        # train and keep adding new experiences to memory
        for e in range(self.numEpisodes):
            done = False
            episodeReward = 0
            state = self.reset_episode();
            while not done:
                # choose action selected by local actor netowrk
                action = self.act(state)
                # take a step, add to memory and train using one sample from memory
                state, episodeReward = self.step(state, action, episodeReward);
                total_steps+=1

            rList.append(episodeReward)
            if len(rList) % 20 == 0 and len(rList) > 0:
                print("total_steps", total_steps, "mean reward", np.mean(rList[-10:]))

        print("finished training")



class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state =self.mu


    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # larger sigma means more change in state , more exploratory
        # randn: Return a sample (or samples) from the “standard normal” distribution.
        # OU process :
        # 1. It essentially generates random samples from a Gaussian (Normal) distribution,
        # but each sample affects the next one such that two consecutive samples are more likely to be closer together
        # than further apart. In this sense, the process in Markovian in nature.

        # since our actions translate to force and torque being applied to a quadcopter,
        # we want consecutive actions to not vary wildly

        # 2. tends to settle down close to the specified mean (self.mu) over time.
        # When used to generate noise, we can specify a mean of zero, and that will have the effect of
        # reducing exploration as we make progress on learning the task.
        # dx is a factor of theta times the difference between x and mean (self.mu)
        # x + dx brings x closer to mu by a factor of theta and random perterbation self.sigma*np.random.randn(len(x))
        # state x will be close to self.state eventually
        dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(len(x))
        self.state = x + dx
        return self.state


def main(args):
    # prepare variables related to the environment
    gymEnv = gym.make('Pendulum-v0')

    env = gym.make('Pendulum-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound_high = env.action_space.high[0]
    action_bound_low = env.action_space.low[0]

    task = namedtuple("myTask", field_names=["state_size", "action_size", "action_low", "action_high"])
    ddpgTask = task(state_size,action_size, action_bound_low, action_bound_high)

    ddpgAgent = DDPG(ddpgTask, env)
    print("start ddpg train")
    ddpgAgent.train()


    return

if __name__ == '__main__':
    main(sys.argv)






