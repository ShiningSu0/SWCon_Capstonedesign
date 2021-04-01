import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd

def flip(edge, np_random):
    return 1 if np_random.uniform() < edge else -1


class PortfolioEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,  edge=0.6, max_wealth=250.0, max_rounds=300):
      self.action_space = spaces.Discrete(int(max_wealth * 100))  # betting in penny
      #할 수 있는 경우의 수가 Discrete( ) 안에 들어감!
      # increments
      self.observation_space = spaces.Tuple((
          spaces.Box(0, max_wealth, [1], dtype=np.float32),  # (w,b)
          spaces.Discrete(max_rounds + 1))) # https://stackoverflow.com/questions/57583185/what-does-spaces-discrete-mean-in-openai-gym
      self.reward_range = (0, max_wealth)
      self.edge = edge
      self.wealth = 100000
      self.initial_wealth = 100000 #Starts at $100,000
      self.max_rounds = max_rounds
      self.max_wealth = max_wealth
      self.np_random = None
      self.rounds = None
      self.seed()
      self.reset()

  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

  def step(self, action):
      bet_in_dollars = min(action / 100.0, self.wealth)  # action = desired bet in pennies
      self.rounds -= 1

      coinflip = flip(self.edge, self.np_random)
      self.wealth = min(self.max_wealth, self.wealth + coinflip * bet_in_dollars)

      done = self.wealth < 0.01 or self.wealth == self.max_wealth or not self.rounds
      reward = self.wealth if done else 0.0

      return self._get_obs(), reward, done, {}

  def _get_obs(self):
      return np.array([self.wealth]), self.rounds

  def reset(self):
      self.rounds = self.max_rounds
      self.wealth = self.initial_wealth
      return self._get_obs()

  def render(self, mode='human'):
      print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)