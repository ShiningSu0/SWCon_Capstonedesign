import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
#https://engineering-ladder.tistory.com/61 구조에 대한 한국어 설명
#https://github.com/hackthemarket/gym-trading/tree/master/gym_trading/envs 참고
def flip(edge, np_random): #이게 이제 주사위가 아니고 투자결과를 return해야 함
    return 1 if np_random.uniform() < edge else -1

#환경 구성
""" time window=300으로 연속적으로 보여줌..?
현재 포트폴리오 비중
현재 자산 가치
30일 치 T10Y2Y의 요약된 결과
Fear & Greed Index
Gold, SPY, QQQ, US T30Y의 RSI
Gold, SPY, QQQ, US T30Y의 STDDEV
"""
class PortfolioEnv(gym.Env):
  metadata = {'render.modes': ['human']}

    def __init__(self,  edge=0.6, max_wealth=200000, max_rounds=300):
      #Action 정의
      self.action_space = spaces.Box(np.array([-1,-1,-1,-1]),np.array([1,1,1,1]))
      # gold, SPY(S&P500), QQQ(NASDAQ),Arbitrage(US T30Y yield)
      #할 수 있는 경우의 수가 Box( ) 안에 들어감!
      # increments
      self.observation_space = spaces.Tuple((
          spaces.Box(np.array([0,0,0,0,0]),np.array([1,1,1,1,1])), #현재 포트폴리오 비중
          spaces.Box(0, max_wealth, shape=[1], dtype=np.float32), # 현재 자산가치
          spaces.Box(n일 치 T10Y2Y),spaces.Box(n일 치 거시경제지표))) # 지표들
          #지표 어떤 거 넣지? T10Y2Y 요약된 결과, 투자시점의 RSI(추세), 변동성(표준편차)
      #data : fred, yahoo finance
          # box는 실수형, discrete는 이산형 범위
          #spaces.Discrete(max_rounds + 1),

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

    def step(self, action):#step 함수를 이용해 에이전트가 환경에 대한 행동 취하고, 이후 획득한 환경에 대한 정보 리턴
      #bet_in_dollars는 포트폴리오의 비중 조정 결정으로 수정되어야.
      bet_in_dollars = min(action / 100.0, self.wealth)  # action = desired bet in pennies
      self.rounds -= 1

      coinflip = flip(self.edge, self.np_random)
      #coinflip은 필요없나..?

      self.wealth = min(self.max_wealth, self.wealth + coinflip * bet_in_dollars)
      # wealth는 포트폴리오의 최대 결과로 수정되어야.

      #done = episode의 종료 여부
      done = self.wealth < 0.01 or self.wealth == self.max_wealth or not self.rounds
      #reward는 수익으로로
      reward = self.wealth if done else 0.0

      return self._get_obs(), reward, done, {}
      #state 이동하면서 얻은 보상을 합치고 기대값을 계산하는 게 Q-Function(Action-Value function)
      #아직은 설정 안했음
    def _get_obs(self): #에이전트가 관찰한 환경 정보를 반환.
      return np.array([self.wealth]), self.rounds

    def reset(self):# Step을 실행하다가 epsiode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 State를 반환한다.
      self.rounds = self.max_rounds
      self.wealth = self.initial_wealth
      return self._get_obs()

    def render(self, mode='human'):
      print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)