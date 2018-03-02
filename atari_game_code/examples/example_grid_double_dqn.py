from __future__ import division

import os
import sys
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(repo_path)
sys.path.insert(0, os.path.join(repo_path, 'agents'))

from gridworld import gameEnv
from duel_DQN import *
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os



env = gameEnv(partial=False,size=5)
print(env.actions) # 4 actions 
# uncomment to see an intial state of the environment
# plt.show()

duelDQN()