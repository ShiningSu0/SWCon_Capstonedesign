import gym
import collections
import random
import gym
import numpy as np
from gym_portfolio.envs.portfolio_env import PortfolioEnv
import pandas as pd
import math
import random
import seaborn as sns
from collections import namedtuple
from itertools import count
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
# Hyperparameters
import torch

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

plt.ion()
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,80)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 79)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10): # 1 에피소드 끝날때마다 train함수 호출시켜서 10번 업데이트함
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        #for loop 돌면서 리스트에 어펜드하고 텐서로 만드는 과정
        q_out = q(s) # 배치처리 일어나는데 q(s)는 사실 s는 32개의 state임 32*4임 왜 32냐면 배치사이즈가 32였음 그래서 32개있음
        #q(s)는 32컴마2임
        q_a = q_out.gather(1, a) # a는 32컴마1 거기있는애들만 골라주는거?
        # q_out은 양쪽 액션의 q가 있는데 실제로 취한액션(a)만 뽑아내야함 즉 취한 액션의 q값만 골랴냄 1은 [32][2] 에서 [2]기준으로 액션 고르란얘기


        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        target=target.float()
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


def main():
    env = gym.make('Portfolio-v0').unwrapped
    q = Qnet()
   #q=q.to('cuda:0')
    q_target = Qnet()
   # q_target=q_target.to('cuda:0')
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    forplot=[]
    barplot=[0]*81
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # Q-net만 업데이트함 큐타겟은 그냥 복사해오니까

    for n_epi in range(300000):
        epsilon = max(0.05, 0.20 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        #10000개 에피소드. 20% 시작해서 1%까지 줄어듦 입실론이 즉 익스플로러 덜하도록
        #액션은 q. sample action
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            barplot[a]+=1
            s_prime, r, done, info = env.step(a) #iloc 10000 넘어갈때 에러 발생 4번마다 리셋해주는 함수 필요함
            done_mask = 0.0 if done else 1.0 # 게임이 끝나는 스텝이면 0 아님 1
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
              #  print("epi : ",n_epi, "memory_size : ",memory.size())
                break

        if memory.size() > 2000: #메모리 2000 전에는 쌓기만 해라.. 랜덤히 움직임 거의
            train(q, q_target, memory, optimizer)
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            if n_epi % (1000*print_interval) == 0:
                forplot.append(score/print_interval)
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    print("l = ",forplot)
    print("barplot = ",barplot)
    env.close()


if __name__ == '__main__':
    main()