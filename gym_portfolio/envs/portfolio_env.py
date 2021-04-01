import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[PortfolioEnv].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class PortfolioEnv(gym.Env, utils.EzPickle):