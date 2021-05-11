import time
import random
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter, writer

class ActorCritic(nn.Module):
    def __init__(self, inputs, outputs):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32*6*6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, outputs)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)
    
    def foward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0,-1)), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx

class A3C(object):
    """Actor critic A3C algorithm from deepmind paper https://arxiv.org/pdf/1602.01783.pdf"""
    def __init__(self, config, actor, critic):
        self.num_processes = multiprocessing.cpu_count()
        self.work_processes = max(1, self.num_processes - 2)
        self.actor_critic = 

class Actor_Critic_Worker(torch.multiprocessing.Process):
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episode_to_run, epsilon_decay_denominator, action_size, action_types, result_queue,
                 local_model, gradient_updates_queue):
        super(Actor_Critic_Worker, self).__init__()
        self.environment = environment
        self.config = config
        self.worker_num = worker_num

        self.gradient_clipping_norm = self.config.hyperparameters["gradient_clipping_norm"]
        self.discount = self.config.hyperparameters["discount"]
        self.normalize_rewards = self.config.hyperparameters["normalise_reward"]

        self.action_size = action_size
        self.set_seeds(self.worker_num)

        self.shared_model = shared_model
        self.local_model = local_model
        self.local_optimizer = shared_optimizer

        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer

        self.episode_to_run = episode_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.hyperparameters["exploration_worker_difference"]
        self.action_types = action_types
        self.result_queue = result_queue
        self.episode_number = 0

        self.gradient_updates_queue = gradient_updates_queue
    
    def set_seeds(self, worker_num):
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)

    def copy_model_over(self):
        for self.local_model, self.shared_modelh in zip(self.local_model.parameters(), self.shared_model.parameters()):
            self.local_model.data.copy(self.shared_model.data.clone())

    def run(self):
        torch.set_num_threads(1)
        for ep_ix in range(self.episode_to_run):
            done = False
            with self.optimizer_lock:
                self.copy_model_over()
            with self.counter.get_lock():
                epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
                epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference, epsilon * self.exploration_worker_difference))
            
            state = self.environment.reset()

            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_action_probabilities = []
            self.critic_outputs = [] 

            while not done:
                action, action_log_prob, critic_outputs =           
            
        