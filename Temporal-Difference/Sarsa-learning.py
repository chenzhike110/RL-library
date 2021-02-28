import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import random
import gym
import time

state_divide_1 = 300
state_divide_2 = 200

class DiscretePolicy(object):
    # using Discrete method to solve continuous problem 
    # this one stick to MountainCar-v0 which has discrete action_space and continuous states
    # MountainCar has 3 action-choice postion between [-1.2 0.6] velocity between [-0.07 0.07]
    # divide position into 360 area, velocity into 280 area, 360*280*3 parameters
    # epsilon determines the probability to take action by weights

    def __init__(self, epsilon, temperature):
        self.states_num = state_divide_1*state_divide_2
        self.action_space = [-1, 0, 1]
        self.action_num = 3
        self.Q = np.zeros((self.states_num, self.action_num))
        # self.Q[:,2] = np.ones(self.states_num)*0.001
        self.epsilon = epsilon
        self.minepsilon = 0.05
        self.alpha = 1e-2
        self.gamma = 0.999
    
    def get_action(self, state):
        num_states = int((state[0] + 1.2)/1.8*(state_divide_1-1))*state_divide_2 + int((state[1] + 0.7)/1.4*(state_divide_2-1))
        if random.random() < self.epsilon:
            action = choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[num_states])]
        return [action]
    
    def update_policy(self, action, state, reward, next_state=None, next_action=None):
        # new V = V + alpha*(reward + gamma*Q_next - V)
        num_states = int((state[0] + 1.2)/1.8*(state_divide_1-1))*state_divide_2 + int((state[1] + 0.7)/1.4*(state_divide_2-1))
        if next_state is not None:
            next_states = int((next_state[0] + 1.2)/1.8*(state_divide_1-1))*state_divide_2 + int((next_state[1] + 0.7)/1.4*(state_divide_2-1))
        current = self.Q[num_states][action+1]
        Q_next = self.Q[next_states][next_action+1] if next_state is not None else 0
        self.Q[num_states][action+1] = current + self.alpha*(reward + self.gamma*Q_next)

def train(env, policy, epoch_num, lr):
    # use montecarlo method to train a policy
    record = []
    for i in range(epoch_num):
        print("czk::gogogo::start training epoch", i)
        done = False
        r_tmp = []
        state = env.reset()
        a = policy.get_action(state)
        while done == False:
            observation, reward, done, info = env.step(a)
            env.render()
            time.sleep(0.005)
            # print("state",observation,"action",a[0],"reward",reward, "info",info)
            r_tmp.append(reward)
            if not done:
                next_action = policy.get_action(observation)
                policy.update_policy(a[0],state,reward,observation,next_action[0])
                state = observation
                a = next_action
            else:
                policy.update_policy(a[0],state,reward)
        policy.epsilon = 2/(i+1)
        record.append(np.sum(r_tmp))
        print("epoch",i,"total rewards",record[-1])
    return record

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    policy = DiscretePolicy(1, 1000)
    total_return = train(env, policy, 500, 1e-3)

    plt.plot(total_return)
    plt.show()

    state = env.reset()
    done = False
    while done == False:
        a = policy.get_action(state)
        observation, reward, done, info = env.step(a)
        env.render()
    env.close()
    
