import gym
import numpy as np
from gym_portfolio.envs.portfolio_env import PortfolioEnv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
import policy_gradient
import tensorflow as tf
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
env = gym.make('Portfolio-v0')
observation = env.reset()
done = False
navs = []
#Executing 1 Episode
while not done:
    action = [0,0,0,1] # stay flat
    observation, reward, done = env.step(action)
    #navs.append(info['nav'])
    if done:
        print('end')