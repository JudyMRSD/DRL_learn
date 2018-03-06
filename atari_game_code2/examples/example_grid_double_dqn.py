from __future__ import division

import os
import sys
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(repo_path)
sys.path.insert(0, os.path.join(repo_path, 'agents'))
sys.path.insert(0, os.path.join(repo_path, 'environment'))
sys.path.insert(0, os.path.join(repo_path, 'runners'))

from gridworld import gameEnv
from duel_DQN import *
from duel_DQN_runner import *

env = gameEnv(partial=False,size=5)
print(env.actions) # 4 actions 
# uncomment to see an intial state of the environment
# plt.show()

# Setting the session to allow growth, so it doesn't allocate all GPU memory.
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

# Setting this as the default tensorflow session.
keras.backend.tensorflow_backend.set_session(sess)
runner = duel_DQN_runner()
print ("train dueling dqn")
# train_writer = tf.summary.FileWriter('../log/', sess.graph)


runner.train(sess)

train_writer.close()








