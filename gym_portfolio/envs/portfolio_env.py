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
# 나스닥, 금, 채권, 애플
#환경 구성
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

action_array_list = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                if i == 0 and j == 0 and k == 0 and l == 0:
                    continue
                else:
                    total_sum = (i + j + k + l)
                    action_array_list.append([i / total_sum, j / total_sum, k / total_sum, l / total_sum])


def get_reward(action,array):#수익을 보상으로써 반환, start_value와 end_value는 각각
    mkt_money_changes_t=100
    money=100
    reward=0
    action_array=action_array_list[action]
    return_array=[]
    mkt_return=[] #시장 수익률
    pf_return=[] #포트폴리오의 일별 수익률
    mkt_money_changes=[]#시장(나스닥)의 일별 수익에 따른 금액 변화 (초기 : 100)
    pf_money_changes=[] #포트폴리오의 일별 수익에 따른 금액 변화 (초기 : 100)
    r_f=array.iloc[60].values[2] #채권수익률 risk_free
    for i in range(0,60,5):#0,60,5인 이유 : cov, var로 베타 계산할 때 data point가 일주일마다 수익률 변화임..
        temp_return=[]
        for j in range(4):
            if j==2:
                temp_return.append((-1)*100 * (array.iloc[i + 5].values[j] - array.iloc[i].values[j]) / array.iloc[i].values[j]) #채권수익률은 반대로 계산
            else:
                temp_return.append(100*(array.iloc[i+5].values[j]-array.iloc[i].values[j])/array.iloc[i].values[j])
        return_array.append(temp_return)

    for i in range(12):#60거래일간의 데이터를 5거래일마다 끊어서 수익률 관찰하므로 12번 반복.
        temp=0
        for j in range(4):
            temp+=action_array[j]*return_array[i][j]
        pf_return.append(temp)
        money=money*((100+temp)/100)
        pf_money_changes.append(money)
        mkt_money_changes_t=mkt_money_changes_t*((100+return_array[i][0])/100)
        mkt_return.append(return_array[i][0])
        mkt_money_changes.append(mkt_money_changes_t)

    beta_pf=np.cov(pf_return,mkt_return)[0, 1]/np.var(mkt_return) #포트폴리오의 베타 계산
    reward=(pf_money_changes[-1]-100)-(r_f+beta_pf*(mkt_money_changes[-1]-100-r_f)) #reward : 수익률 - CAPM에 따른 기대 수익률 -> market excess return?




    return reward





class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,  max_wealth=2000000, days=240): #max 200000 : 100000에서 2배까지 불어나면 종료
      #Action 정의
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
    def step(self, action):#step 함수를 이용해 에이전트가 환경에 대한 행동 취하고, 이후 획득한 환경에 대한 정보 리턴
      self.idx += 60
      self.stepcount += 1
      self.done = (self.stepcount >= 4)
     # print("시도 : ", self.stepcount, "현재 액션 : ",action)
      reward = get_reward(action,self.data.iloc[self.idx:self.idx+61])
      self.portfolio_proportion=0
      self.wealth += reward
   #   print("portfolio_env 클래스에서 step 함수에서 현재자산가치 :", self.wealth)
      trend=self.data.iloc[self.idx+120][1]-self.data.iloc[self.idx+60][1]
      return np.array([self.wealth,self.indicators.iloc[self.idx + 120][0],self.indicators.iloc[self.idx + 120][1],self.indicators.iloc[self.idx + 120][2]],dtype=np.float32), reward,self.done,trend#self.get_observation(), reward,self.done

    def reset(self):# Step을 실행하다가 epsiode가 끝나서 이를 초기화해서 재시작해야할 때, 초기 State를 반환한다.
      self.wealth = self.initial_wealth
      self.stepcount=0
      self.done=0
      self.idx= np.random.randint(low=0, high=len(self.data.index) -360)

      return np.array([self.wealth,self.indicators.iloc[self.idx + 60][0],self.indicators.iloc[self.idx + 60][1],self.indicators.iloc[self.idx + 60][2]],dtype=np.float32)#self._get_obs()
