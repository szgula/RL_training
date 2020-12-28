import gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 512)  # 6*6 from image dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self):
        self.memory_s = []
        self.memory_r = []
        self.net = Net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.debug_loss = []

    def get_action(self, state):
        input_ = torch.tensor([*state, 0]).float()
        val_0 = self.net(input_)
        input_[-1] = 1
        val_1 = self.net(input_)
        if np.random.random() < 0.15:
            val_0, val_1 = val_1, val_0
        return int(val_1 > val_0)

    def add_memory(self, state, action, reward):
        self.memory_s.append(np.array([*state, action]))
        self.memory_r.append(reward)

    def update_net(self, total_reward):
        #  update action value function
        self.net.zero_grad()
        inputs = torch.tensor(self.memory_s).float()
        rew_disc = [0]
        for i, v in enumerate(self.memory_r):
            rew_disc.append(rew_disc[-1] * 0.99 + v)
        rew_disc.pop(0)
        rew_disc.reverse()
        cum_reward_to_end = torch.tensor(rew_disc).float()
        #cum_reward_to_end = torch.tensor(total_reward - np.cumsum(self.memory_r)).float().clip(0, 10)
        output = self.net(inputs)
        loss = self.criterion(output, cum_reward_to_end.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        output_new = self.net(inputs)
        self.debug_loss.append(loss.item())
        self.memory_r = []
        self.memory_s = []


class SimpleAgent:
    def __init__(self):
        self.memory_s = []
        self.memory_r = []
        self.action_value = np.ones((50, 200, 50, 200, 2)) * 100

    def get_indexes(self, state):
        id_0 = min(max(int(state[0] * 5) + 25, 0), 49)
        id_1 = min(max(int(state[1] * 10) + 50, 0), 99)
        id_2 = min(max(int(state[2] * 50) + 25, 0), 49)
        id_3 = min(max(int(state[3] * 10) + 50, 0), 99)
        if len(state) == 5:
            return id_0, id_1, id_2, id_3, int(state[4])
        return id_0, id_1, id_2, id_3

    def get_action(self, state):
        indexes = self.get_indexes(state)
        return self.action_value[indexes].argmax()

    def add_memory(self, state, action, reward):
        self.memory_s.append(np.array([*state, action]))
        self.memory_r.append(reward)

    def update_net(self, total_reward):
        cum_sum = 0
        for s, r in zip(reversed(self.memory_s), reversed(self.memory_r)):
            cum_sum = 0.99 * cum_sum + r
            idx = self.get_indexes(s)
            self.action_value[idx] = (0.1 * cum_sum) + (0.9 * self.action_value[idx])
        self.memory_s = []
        self.memory_r = []




if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    len_ = []
    my_agent = Agent()
    #my_agent = SimpleAgent()
    for i_episode in range(10000):
        observation = env.reset()
        cum_rew = 0
        for t in range(500):
            if i_episode % 100 == 0:
                env.render()
            #print(observation)
            #act = env.action_space.sample()
            act = my_agent.get_action(observation)
            new_observation, rew, done, info = env.step(act)
            cum_rew += rew
            my_agent.add_memory(observation, act, rew)
            observation = new_observation
            if done:
                if i_episode % 50 == 0:
                    print(f"Episode {i_episode} finished after {t+1} timesteps")
                len_.append(t+1)
                break
        my_agent.update_net(cum_rew)

    for t in range(100):
        env.render()
        act = my_agent.get_action(observation)
        new_observation, rew, done, info = env.step(act)
        if done: break

    env.close()
    #plt.plot(my_agent.debug_loss)
    #plt.savefig('loss.png')
    #plt.close()
    plt.plot(len_)
    plt.savefig('length.png')
