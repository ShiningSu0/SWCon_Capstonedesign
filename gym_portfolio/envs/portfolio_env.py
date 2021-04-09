import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
#https://engineering-ladder.tistory.com/61 구조에 대한 한국어 설명
#https://github.com/hackthemarket/gym-trading/tree/master/gym_trading/envs 참고

#환경 구성
def reward(action):

class PortfolioEnv(gym.Env):
  metadata = {'render.modes': ['human']}

    def __init__(self,  max_wealth=200000, days=400): #max 200000 : 100000에서 2배까지 불어나면 종료
      #Action 정의
      self.action_space = spaces.Box(np.array([-1,-1,-1,-1]),np.array([1,1,1,1]))
      # gold, SPY(S&P500), QQQ(NASDAQ),Arbitrage(US T30Y yield)
      #할 수 있는 경우의 수가 Box( ) 안에 들어감! datas
      # increments
      #https: // assethorizon.tistory.com / 18
      #https://medium.com/swlh/states-observation-and-action-spaces-in-reinforcement-learning-569a30a8d2a1
      self.observation_space = spaces.Tuple((
          spaces.Box(np.array([0,0,0,0]),np.array([1,1,1,1])), #현재 포트폴리오 비중
          spaces.Box(0, max_wealth, shape=[1], dtype=np.float32), # 현재 자산가치
          spaces.Box(shape=(100,4),dtype=np.float32),#[NASDAQ,DOWJONES,GOLD,DGS30]
          spaces.Box(shape=(1,3),dtype=np.float32)))#indicators[Spread(T10Y-2Y),RSI,Volatility(Var)]
      #data : fred, yahoo finance
          #반년 치 보여주고, 투자하고, 반년 치 보여주고, 또 투자하고.. 총 3년
          # box는 실수형, discrete는 이산형 범위
          #spaces.Discrete(days + 1),
      self.data=pd.read_csv('data.csv',index_col=0)
      self.indicators=pd.read_csv('indicators.csv',index_col=0)
      self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days)
      self.reward_range = (0, max_wealth)
      self.stepcount=0
      self.wealth = 100000
      self.initial_wealth = 100000 #Starts at $100,000
      self.portfolio_proportion=[0,0,0,1] #비중의 초기값 NASDAQ, DOWJONES,GOLD,DGS30(Arbitrage)
      self.days = days
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
        self.data.iloc[self.idx:self.idx+101].values,#[self.idx:self.idx+30]도 가능 이부분 인덱스 잘 맞춰줘야
        self.data.iloc[self.idx + 100].values # 마지막 시점에서 요약된 indicator들을 보여줌.
                   )
      self.idx += 100
      self.stepcount += 1
      done = self.stepcount >= 4
      #??yret = observation[2]

      reward, info = self.sim._step(action, yret)

      # info = { 'pnl': daypnl, 'nav':self.nav, 'costs':costs }

      return observation, reward, done, info

      """#bet_in_dollars는 포트폴리오의 비중 조정 결정으로 수정되어야.
      bet_in_dollars = min(action / 100.0, self.wealth)  # action = desired bet in pennies
      self.rounds -= 1

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
      return np.array([self.wealth]), self.rounds"""

    def reset(self):# Step을 실행하다가 epsiode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 State를 반환한다.
      self.wealth = self.initial_wealth
      self.stepcount=0
      return self._get_obs()

    def render(self, mode='human'):
      print("Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)