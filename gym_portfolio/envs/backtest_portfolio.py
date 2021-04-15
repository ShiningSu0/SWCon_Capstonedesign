import gym
import numpy as np
from gym_portfolio.envs.portfolio_env import PortfolioEnv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive

env = gym.make('Portfolio-v0')
observation = env.reset()
done = False
navs = []
while not done:
    action = 1 # stay flat
    observation, reward, done = env.step(action)
    #navs.append(info['nav'])
    if done:
        print ('Annualized return: ')#,navs[len(navs)-1]-1)
        #pd.DataFrame(navs).plot()