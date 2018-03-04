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

runner = duel_DQN_runner()
runner.train()
