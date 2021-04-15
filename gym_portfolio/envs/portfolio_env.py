import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
#https://engineering-ladder.tistory.com/61 구조에 대한 한국어 설명
#https://github.com/hackthemarket/gym-trading/tree/master/gym_trading/envs 참고

#환경 구성
def get_reward(action,start_value,end_value):
  reward=0
  return reward

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,  max_wealth=200000, days=240): #max 200000 : 100000에서 2배까지 불어나면 종료
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
      self.idx = np.random.randint(low=0, high=len(self.data.index) - days-60)
      self.reward_range = (0, max_wealth)
      self.stepcount=0
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

    def step(self, action):#step 함수를 이용해 에이전트가 환경에 대한 행동 취하고, 이후 획득한 환경에 대한 정보 리턴
      observation=(
        self.portfolio_proportion,
        self.wealth,
        self.data.iloc[self.idx:self.idx+61].values,#[self.idx:self.idx+30]도 가능 이부분 인덱스 잘 맞춰줘야
        self.indicators.iloc[self.idx + 60].values # 마지막 시점에서 요약된 indicator들을 보여줌.
                   )
      self.idx += 60
      self.stepcount += 1
      done = self.stepcount >= 4

      reward = get_reward(action,self.data.iloc[self.idx+60].values,self.data.iloc[self.idx+120].values)
      self.wealth += reward
      return observation, reward, done

    def reset(self):# Step을 실행하다가 epsiode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 State를 반환한다.
      self.wealth = self.initial_wealth
      self.stepcount=0
      return 0#self._get_obs()

    def render(self, mode='human'):
      print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)