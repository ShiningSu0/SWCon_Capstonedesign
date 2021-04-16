import gym
import numpy as np
from gym_portfolio.envs.portfolio_env import PortfolioEnv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
import policy_gradient


env = gym.make('Portfolio-v0')
observation = env.reset()
done = False
navs = []
while not done:
    action = [1,0,0,0] # stay flat
    observation, reward, done = env.step(action)
    #navs.append(info['nav'])
    if done:
        print('end')#,navs[len(navs)-1]-1)
        #pd.DataFrame(navs).plot()

"""

# create the tf session
sess = tf.InteractiveSession()

# create policygradient
pg = policy_gradient.PolicyGradient(sess, obs_dim=5, num_actions=3, learning_rate=1e-2 )

# and now let's train it and evaluate its progress.  NB: this could take some time...
df,sf = pg.train_model( env,episodes=25001, log_freq=100)#, load_model=True)

sf['net'] = sf.simror - sf.mktror
#sf.net.plot()
sf.net.expanding().mean().plot()
sf.net.rolling(100).mean().plot()
"""