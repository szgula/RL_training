import gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Temporal Difference: TD(0) for action-value function
1) TabularSarsa - based on the tables (basic tile coding)
2) ANNSarsa - based on the NN represenation
"""


class TabularSarsa:
    def __init__(self, quantization_tuple: tuple = (50, 200, 50, 200, 2), state_scalars: tuple = (5, 10, 50, 10),
                 gamma: float = 1, alpha: float = 0.4, epsilon: float = 0.2):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.quantization_tuple = quantization_tuple
        self.state_scalars = state_scalars
        self.action_value = np.ones(self.quantization_tuple) * 0

    def get_indexes(self, state, action=None):
        q_0, q_1, q_2, q_3, _ = self.quantization_tuple
        s_0, s_1, s_2, s_3 = self.state_scalars
        id_0 = min(max(int(state[0] * s_0) + 25, 0), q_0-1)
        id_1 = min(max(int(state[1] * s_1) + 50, 0), q_1-1)
        id_2 = min(max(int(state[2] * s_2) + 25, 0), q_2-1)
        id_3 = min(max(int(state[3] * s_3) + 50, 0), q_3-1)
        if action is None:
            return id_0, id_1, id_2, id_3
        return id_0, id_1, id_2, id_3, int(action)

    def get_action(self, state, enable_exploration=True):
        indexes = self.get_indexes(state)
        if np.random.random(1) < self.epsilon and enable_exploration:
            return self.action_value[indexes].argmin()
        return self.action_value[indexes].argmax()

    def update(self, state, action, reward, next_state, next_action):
        """
        Policy evaluation based on the SARSA (Generalized Policy Iteration)
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param next_action:
        :return:
        """
        idx0 = self.get_indexes(state, action)
        idx1 = self.get_indexes(next_state, next_action)
        delta = reward + self.gamma * self.action_value[idx1] - self.action_value[idx0]
        self.action_value[idx0] += self.alpha * delta


class TabularSarsaNSteps(TabularSarsa):
    def __init__(self, *args, delay_steps:int = 3, **kwargs):
        super().__init__(*args, gamma=1,**kwargs)
        self.delay_steps = delay_steps
        self.states = deque(maxlen=delay_steps)
        self.rewards = deque(maxlen=delay_steps)
        self.actions = deque(maxlen=delay_steps)

    def update(self, state, action, reward, done_, _internal_call=False):
        """
        Policy evaluation based on the SARSA (Generalized Policy Iteration)
        :param _internal_call:
        :param done_:
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param next_action:
        :return:
        """
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        if len(self.states) < self.delay_steps: return
        idx0 = self.get_indexes(self.states[0], self.actions[0])
        idx1 = self.get_indexes(self.states[-1], self.actions[-1])
        cum_reward = sum(self.rewards) - self.rewards[-1]
        future_reward = self.action_value[idx1] if not done_ else 0
        delta = cum_reward + future_reward - self.action_value[idx0]
        self.action_value[idx0] += self.alpha * delta
        if done_ and not _internal_call:
            for _ in range(self.delay_steps - 1):
                self.update(state, action, 0, done_, _internal_call=True)
            self.states = deque(maxlen=self.delay_steps)
            self.rewards = deque(maxlen=self.delay_steps)
            self.actions = deque(maxlen=self.delay_steps)


class NNSarsa:
    def __init__(self, gamma: float = 1, alpha: float = 0.4, epsilon: float = 0.2):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.net = Net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())#, lr=self.alpha)

    def get_action(self, state, enable_exploration=True):
        input_ = torch.tensor([*state, 0]).float()
        val_0 = self.net(input_)
        input_[-1] = 1
        val_1 = self.net(input_)
        if np.random.random() < self.epsilon and enable_exploration:
            val_0, val_1 = val_1, val_0
        return int(val_1 > val_0)

    def update(self, state, action, reward, done, next_state, next_action):
        self.net.zero_grad()
        input_0 = np.array([*state, action])
        input_0_torch = torch.tensor(input_0).float()
        prediction_0 = self.net(input_0_torch)
        input_1 = np.array([*next_state, next_action])
        input_1_torch = torch.tensor(input_1).float()
        expected = reward + self.gamma * self.net(input_1_torch).float()
        loss = self.criterion(prediction_0, expected)
        loss.backward()
        pass


class NNSarsaNStep:
    def __init__(self, gamma: float = 1, alpha: float = 0.4, epsilon: float = 0.2, delay_steps: int = 10):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.net = Net()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.net.parameters())#, lr=self.alpha)
        self.delay_steps = delay_steps
        self.states = deque(maxlen=delay_steps)
        self.rewards = deque(maxlen=delay_steps)
        self.actions = deque(maxlen=delay_steps)

    def get_action(self, state, enable_exploration=True):
        input_ = torch.tensor([*state, 0]).float()
        val_0 = self.net(input_)
        input_[-1] = 1
        val_1 = self.net(input_)
        if np.random.random() < self.epsilon and enable_exploration:
            val_0, val_1 = val_1, val_0
        return int(val_1 > val_0)

    def update(self, state, action, reward, done_, _internal_call=False):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        if len(self.states) < self.delay_steps: return
        cum_reward = sum(self.rewards) - self.rewards[-1]

        input_0 = np.array([*self.states[0], self.actions[0]])
        input_0_torch = torch.tensor(input_0).float()
        prediction_0 = self.net(input_0_torch)

        input_1 = np.array([*state, action])
        input_1_torch = torch.tensor(input_1).float()
        future_reward = self.net(input_1_torch).float()
        if done: future_reward *= 0
        expected = cum_reward + future_reward
        loss = self.criterion(prediction_0, expected)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        if done_ and not _internal_call:
            for _ in range(self.delay_steps - 1):
                self.update(state, action, 0, done_, _internal_call=True)
            self.states = deque(maxlen=self.delay_steps)
            self.rewards = deque(maxlen=self.delay_steps)
            self.actions = deque(maxlen=self.delay_steps)


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



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    len_ = []
    #my_agent = TabularSarsaNSteps(delay_steps=10, alpha=0.1)
    my_agent = NNSarsaNStep(delay_steps=3)  #NNSarsa()
    for i_episode in range(4000):
        observation = env.reset()
        act = my_agent.get_action(observation)
        cum_rew = 0
        for t in range(500):
            if i_episode % 100 == 0: env.render()
            enable_exploration = i_episode % 1000 != 0
            new_observation, rew, done, info = env.step(act)
            if done: rew = 0
            next_act = my_agent.get_action(observation, enable_exploration=enable_exploration)
            my_agent.update(observation, act, rew, done)
            #my_agent.update(observation, act, rew, done, new_observation, next_act)
            observation, act = new_observation, next_act
            if done: break
        if i_episode % 25 == 0:
            print(f"Episode {i_episode} finished after (mean) {np.average(len_[:-100])} timesteps (current {t + 1})")
        len_.append(t + 1)

    observation = env.reset()
    for t in range(1000):
        env.render()
        act = my_agent.get_action(observation, enable_exploration=False)
        observation, rew, done, info = env.step(act)
        if done:
            print(f"Episode LAST finished after {t + 1} timesteps")
            break

    env.close()
    #plt.plot(my_agent.debug_loss)
    #plt.savefig('loss.png')
    #plt.close()
    plt.plot(len_)
    plt.savefig('length.png')
