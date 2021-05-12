import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#https://engineering-ladder.tistory.com/61 구조에 대한 한국어 설명
#https://github.com/hackthemarket/gym-trading/tree/master/gym_trading/envs 참고

#환경 구성
def get_reward(action,wealth,start_value,end_value):#수익을 보상으로써 반환
  reward=0
  action_array=[]
  #print("portfolio_env.py get_reward 함수에서 action : ",action)
  if action==0:
      action_array=[1,0,0,0]
  elif action==1:
      action_array=[0,1,0,0]
  elif action==2:
      action_array=[0,0,1,0]
  elif action==3:
      action_array=[0,0,0,1]
  elif action==4:
      action_array=[0.5,0.5,0,0]
  elif action==5:
      action_array=[1,0,0,0]
  elif action==6:
      action_array=[1,0,0,0]
  elif action==7:
      action_array=[1,0,0,0]
  elif action==8:
      action_array=[1,0,0,0]
  elif action==9:
      action_array=[1,0,0,0]
  elif action==10:
      action_array=[1,0,0,0]
  elif action==11:
      action_array=[1,0,0,0]

  #print("portfolio_env 클래스에서 step 함수에서 비중 :",action_array)
  for i in range(3):
    reward+=((action_array[i]*wealth*(end_value[i]-start_value[i]))/start_value[i])
  #DGS30 채권수익률은 다르게 계산됨.
  reward+=action_array[3]*wealth*((1+0.01*(start_value[3])/4)-1)
  return reward

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,  max_wealth=2000000, days=240): #max 200000 : 100000에서 2배까지 불어나면 종료
      #Action 정의
      self.action_space = spaces.Box(np.array([-1,-1,-1,-1]),np.array([1,1,1,1]))
      #https: // assethorizon.tistory.com / 18
      #https://medium.com/swlh/states-observation-and-action-spaces-in-reinforcement-learning-569a30a8d2a1
      self.observation_space = spaces.Tuple((
          spaces.Box(np.array([0,0,0,0]),np.array([1,1,1,1])), #현재 포트폴리오 비중
          spaces.Box(0, max_wealth, shape=[1], dtype=np.float32), # 현재 자산가치
          spaces.Box(low=-10000,high=100000,shape=(60,4),dtype=np.float32),#[NASDAQ,WILSHIRE,GOLD,DGS30]
          spaces.Box(low=-10000,high=100000,shape=(1,6),dtype=np.float32)))#indicators[Spread(T10Y-2Y),NASDAQ_RSI,WILSHIRE_RSI,GOLD_RSI,DGS30_RSI,Volatility(Var)]
      #data : fred, yahoo finance
          #3개월(60거래일) 치 보여주고, 투자하고, 3개월 치 보여주고, 또 투자하고.. 총 2년
          # box는 실수형, discrete는 이산형 범위
          #spaces.Discrete(days + 1),
      self.data=pd.read_csv('data.csv',index_col=0)
      self.indicators=pd.read_csv('indicators.csv',index_col=0)
      self.idx = np.random.randint(low=0, high=len(self.data.index) - 360)
      self.reward_range = (0, max_wealth)
      self.stepcount=0
      self.done=0
      self.wealth = 100000
      self.initial_wealth = 100000 #Starts at $100,000
      self.portfolio_proportion=[0,0,0,1] #비중의 초기값 NASDAQ, WILSHIRE,GOLD,DGS30(Arbitrage)
      self.max_wealth = max_wealth
      self.np_random = None
      self.seed()
      self.reset()

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
    def get_observation(self):
        a=np.array(self.portfolio_proportion)
        b=np.array([self.wealth,0,0,0])
        c=np.array(self.data.iloc[self.idx:self.idx + 61])
       # d=np.array(self.indicators.iloc[self.idx + 60])
        observation=torch.from_numpy(np.vstack((a,b,c)))
        """observation = (
            self.portfolio_proportion,
            self.wealth,
            self.data.iloc[self.idx:self.idx + 61].values,  # [self.idx:self.idx+30]도 가능 이부분 인덱스 잘 맞춰줘야
            self.indicators.iloc[self.idx + 60].values  # 마지막 시점에서 요약된 indicator들을 보여줌.
        )"""
        return observation
    def step(self, action):#step 함수를 이용해 에이전트가 환경에 대한 행동 취하고, 이후 획득한 환경에 대한 정보 리턴
      self.idx += 60
      self.stepcount += 1
      self.done = (self.stepcount >= 4)
      #print(self.idx)
      reward = get_reward(action,self.wealth,self.data.iloc[self.idx+60].values,self.data.iloc[self.idx+120].values)
      self.portfolio_proportion=0
      self.wealth += reward
   #   print("portfolio_env 클래스에서 step 함수에서 현재자산가치 :", self.wealth)

      return np.array([self.wealth,self.indicators.iloc[self.idx + 120][0],self.indicators.iloc[self.idx + 120][1],self.indicators.iloc[self.idx + 120][2]],dtype=np.float32), reward,self.done,{}#self.get_observation(), reward,self.done

    def reset(self):# Step을 실행하다가 epsiode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 State를 반환한다.
      self.wealth = self.initial_wealth
      self.stepcount=0
      self.done=0
      self.idx= np.random.randint(low=0, high=len(self.data.index) -360)
      #print(self.indicators.iloc[self.idx + 60][0])
      #print(self.idx)
      return np.array([self.wealth,self.indicators.iloc[self.idx + 60][0],self.indicators.iloc[self.idx + 60][1],self.indicators.iloc[self.idx + 60][2]],dtype=np.float32)#self._get_obs()

    def render(self, mode='human'):
      print("Current wealth: ", self.wealth)