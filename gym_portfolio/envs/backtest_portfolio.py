import gym

env = gym.make('CartPole-v0')
for i_episode in range(20):
    # 새로운 에피소드(initial environment)를 불러온다(reset)
    observation = env.reset()
    for t in range(100):
        env.render()
        # 행동(action)을 취하기 이전에 환경에 대해 얻은 관찰값(observation)
        print('observation before action:')
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # 행동(action)을 취한 이후에 환경에 대해 얻은 관찰값(observation)
        print('observation after action:')
        print(observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break