import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

hyperparameters = {
    "device":torch.device("cuda"),
    "hidden_list":[4, 8, 16, 8]
    "lr":1e-3,
    "training_epoch":1000,
    "gamma":0.99,
}

class policy_net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list)
        super(policy_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_list[0]), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_list[0], hidden_list[1]), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_list[1], hidden_list[2]), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(hidden_list[2], hidden_list[3]), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(hidden_list[3], out_dim), nn.ReLU(True),nn.Softmax(out_dim))

    def forward(x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

class Reinforce(object):
    def __init__(self, env):
        self.env = env
        self.policy = policy_net(2, 3, hyperparameters["hidden_list"])
        self.optimizer = optim.Adam(self.policy.parameters, lr=hyperparameters["lr"])
        self.episode_rewards = []
        self.epoch_num = hyperparameters["training_epoch"]
        self.log_probabilities = []
        self.device = hyperparameters["device"]

    def get_action_and_record(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probobility = self.policy(state)
        action_distribution = Categorical(action_probobility)
        action = action_distribution.sample().item()
        self.log_probabilities.append(action_distribution.log_prob(action))
        return action
    
    def train(self):
        state = self.env.reset()
        action = get_action_and_record(state)
        while self.epoch_num > 0:
            next_state, reward, done, _ = self.env.step(action)
            self.episode_rewards.append(reward)
            if done:
                self.update()

    def update(self):
        episode_G = np.zeros_like(self.episode_rewards)
        discounted_return = 0
        for i in range(len(self.episode_rewards)-1,-1,-1):
            discounted_return = discounted_return*hyperparameters["gamma"]+self.episode_rewards[i]
            episode_G[i] = discounted_return
        
        # normalize episode rewards
        episode_G -= np.average(episode_G)
        episode_G /= np.std(episode_G)