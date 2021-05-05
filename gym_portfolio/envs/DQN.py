import gym
import numpy as np
from gym_portfolio.envs.portfolio_env import PortfolioEnv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('Portfolio-v0').unwrapped
#observation = env.reset()
env.reset()
done = False
navs = []
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##Cartpole 알고리즘 차용 https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#https://github.com/AndersonJo/dqn-pytorch/blob/master/dqn.py dqn 참고
class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.linear_1=nn.Linear(self.n_action,4)
        self.linear_2=nn.Linear(4,4)
    def forward(self, input_tensor):
        linear1=F.relu(self.linear_1(input_tensor))
        output=F.sigmoid(self.linear_2(linear1))
        return output
"""
class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        LSTM_MEMORY = 128
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=1, padding=1)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

        self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
        # self.affine2 = nn.Linear(2048, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x, hidden_state, cell_state):
        # CNN
        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state

    def init_states(self) -> [Variable, Variable]:
        LSTM_MEMORY = 128
        hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()
"""
#################################
#학습 모듈
#################################
#다음주 할 일 : 중간보고서 발표, DQN 연결해보기

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.shape[0] #이거 대신 action 수인 4를 넣어야 하나..?

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            print("걸림3")
            return policy_net(state)#.max(1)[1].view(1, 1)
    else:
        print('랜덤')
        #random하고 비중의 합이 1이 되게끔 하는 input이 필요 -> 포트폴리오 비중.
        r1=random.random()
        r2=random.random()
        r3=random.random()
        r4=random.random()
        r_sum=(r1+r2+r3+r4)
        r1=r1/r_sum
        r2 = r2 / r_sum
        r3 = r3 / r_sum
        r4 = r4 / r_sum
        return torch.tensor([r1,r2,r3,r4], device=device, dtype=torch.float64)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


#training

""" 예시 : 한 에피소드만 돌려볼 때
#Executing 1 Episode
while not done:
    action = [0,0,0,1] # stay flat
    observation, reward, done = env.step(action)
    #navs.append(info['nav'])
    if done:
        print('end')
"""

num_episodes = 50
for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    env.reset()
    state=env.get_observation()
    #last_screen = get_screen()
    #current_screen = get_screen()
    #state = current_screen - last_screen

    for t in count():
        # 행동 선택과 수행
        action = select_action(state)
        new_observation, reward, done = env.step(action)
      #  _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 새로운 상태 관찰
       # last_screen = current_screen
       # current_screen = get_screen()
        if not done:
            next_state = new_observation
        else:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # 최적화 한단계 수행(목표 네트워크에서)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    #목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
