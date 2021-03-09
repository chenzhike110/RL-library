import torch
import numpy as np
import random
from collections import namedtuple, deque
import gym
from torch import nn 
import torch.optim as optim
import time

hyperparameters = {
    "memory_size":2500,
    "batch_size":64,
    "lr":1e-2,
    "min_epsilon":0.05,
    "action_space":3,
    "discount_rate":0.99,
    "start_train":300,
    "train_epoch":1000,
}

class replay_buffer(object):

    def __init__(self, max_length, batch_size):
        self.length = max_length
        self.memory = deque(maxlen=self.length)
        self.experience = namedtuple("Experience", field_names=["state","action","reward","next_state","done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def add_experience(self, states, actions, rewards, next_states, dones):
        if len(states)>1:
            assert len(dones) != len(states), "data format error"
            experience = [self.experience(state, action, reward, next_state, done) for 
                            state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experience)
        else:
            experience = self.experience(state, action, reward, next_state, done)
            self.memory.append(experience)
    
    def get_experience(self, num_experience=None, change_format=True):
        if num_experience is not None:
            batch_size = self.batch_size
        else:
            batch_size = num_experience
        experience = random.sample(self.memory, k=batch_size) 
        if change_format:
            states = torch.from_numpy(np.vstack([e.state for e in experience])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experience])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experience])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.rewards for e in experience])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experience])).float().to(self.device)
            return states, actions, rewards, next_states, dones
        else:
            return experience
    
    def __len__(self):
        return len(self.memory)

class Q_network(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super(Q_network, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_list[0]), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_list[0], hidden_list[1]), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_list[1], hidden_list[2]), nn.ReLU(True))
        self.layer4 = nn.Linear(hidden_list[-1],out_dim)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DQN_agent(object):

    def __init__(self, env):
        self.memory = replay_buffer(hyperparameters["memory_size"], hyperparameters["batch_size"])
        self.epsilon = 0
        self.minepsilon = hyperparameters["min_epsilon"]
        self.num_action = hyperparameters["action_space"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network = Q_network(2,self.num_action,[8,16,8]).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hyperparameters["lr"], eps=1e-4)
        self.done = False
        self.remain_epoch = hyperparameters["train_epoch"]
        self.env = env

    def get_action(self, state):
        if type(state) == int:
            state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = random.randrange(0,3)
            return action
        else:
            self.q_network.eval()
            with torch.no_grad():
                action_value = self.q_network(state)
            self.q_network.train()
            action = int(torch.argmax(action_value))
            return action
    
    def learn_from_memory(self):
        states, actions, rewards, next_states, dones = self.memory.get_experience()
        with torch.no_grad():
            Q_targets = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
            Q_target_now = rewards + hyperparameters["discount_rate"] * Q_targets * (1 - dones)
        Q_expected = self.q_network(states).gather(1, actions.long())
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        while self.remain_epoch > 0:
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            done = False
            state = self.env.reset()
            action = self.get_action(state)
            while not done:
                next_state, reward, done, info = self.env.step(action)
                self.env.render()
                time.sleep(0.05)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                state = next_state
                action = self.get_action(state)
                if done and reward > 0:
                    self.memory.add_experience(states, actions, rewards, next_states, dones)
            if len(self.memory)>hyperparameters["start_train"]:
                self.learn_from_memory()
                self.remain_epoch -= 1

    def test(self):
        state = self.env.reset()
        action = self.get_action(state)
        while not done:
            next_state, reward, done, info = env.step(action)
            self.env.render()
            state = next_state
            action = self.get_action(state)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    agent = DQN_agent(env)
    agent.train()
    agent.test()