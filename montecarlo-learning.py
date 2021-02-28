import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import random
import gym
import time

class DiscretePolicy(object):
    # using softmax and Discrete method to solve continuous problem 
    # this one stick to MountainCar-v0 which has discrete action_space and continuous states
    # MountainCar has 3 action-choice postion between [-1.2 0.6] velocity between [-0.07 0.07]
    # divide position into 360 area, velocity into 280 area, 360*280*3 parameters
    # epsilon determines the probability to take action by weights

    def __init__(self, epsilon, temperature):
        self.states_num = 360*280
        self.action_space = [-1, 0, 1]
        self.action_num = 3
        self.weights = np.zeros((self.states_num, self.action_num))
        self.epsilon = epsilon
        self.minepsilon = 0.1
        self.temperature = temperature
        self.mintemperature = 0.5
    
    def get_action(self, state):
        num_states = int((state[0] + 1.2)/1.8*359)*280 + int((state[1] + 0.7)/1.4*279)
        if random.random() < self.epsilon:
            tmp = np.exp(self.weights[num_states])
            prob = tmp / sum(tmp)
            action = choice(self.action_space, p=prob)
        else:
            action = self.action_space[np.argmax(self.weights[num_states])]
        return [action]
    
    def update_policy(self, action, advantage, states, policy_step):
        # loss function = - sum(advantages * ln(yhat))
        # yhat = softmax(weights)
        # if argmax(self.weight(state) == i)
        # gradient_i = advantages * yhat^-1 * (yhat - exp(output_i)^2/sum(exp(output))^2) = (1 - softmax(self.weights(state)[i])) * yreal
        # if not
        # gradient_i = advantages * yhat^-1 * -(exp(output_real)*exp(output_i))/sum(exp(output))^2 = -softmax(self.weights(state)[i]) * yreal
        grad = np.zeros((self.states_num, self.action_num))
        for index, state in enumerate(states):
            num_states = int((state[0] + 1.2)/1.8*359)*280 + int((state[1] + 0.7)/1.4*279)
            tmp = np.exp(self.weights[num_states])
            prob = tmp / sum(tmp) / self.temperature
            for n, act in enumerate(self.action_space):
                if action[index] == act:
                    grad[num_states][n] = (1 - prob[n]) * advantage[index] / self.temperature
                else:
                    grad[num_states][n] = -prob[n] * advantage [index] / self.temperature
        self.temperature -= (self.temperature - self.mintemperature) / 100
        self.epsilon -= (self.epsilon - self.minepsilon) / 100
        self.weights = self.weights + grad * policy_step

class ValueFunction(object):
    def __init__(self, gamma):
        self.states_num = 360*280
        self.action_num = 3
        self.values = np.zeros((self.states_num))
        self.gamma = gamma
    
    def get_value(self,state):
        num_states = int((state[0] + 1.2)/1.8*359)*280 + int((state[1] + 0.7)/1.4*279)
        return self.values[num_states]
    
    def update(self, states, value_estimate, target, value_step):
        for index, state in enumerate(states):
            num_states = int((state[0] + 1.2)/1.8*359)*280 + int((state[1] + 0.7)/1.4*279)
            self.values[num_states] += value_step*(target[index]-value_estimate[index])
    
    def get_discounted_return(self, rewards):
        discounted_return = np.zeros(len(rewards))
        future = 0
        for i in range(len(rewards)-1, -1, -1):
            discounted_return[i] = future * self.gamma + rewards[i]
            future = discounted_return[i]
        return discounted_return

def train(env, policy, Valuefunction, epoch_num, lr):
    # use montecarlo method to train a policy
    record = []
    for i in range(epoch_num):
        print("czk::gogogo::start training epoch", i)
        done = False
        s_tmp = []
        a_tmp = []
        r_tmp = []
        state = env.reset()
        while done == False:
            a = policy.get_action(state)
            observation, reward, done, info = env.step(a)
            env.render()
            time.sleep(0.005)
            # print("state",observation,"action",a[0],"reward",reward, "info",info)
            s_tmp.append(observation)
            a_tmp.append(a[0])
            r_tmp.append(reward)
            if done and reward <= 0 and random.random() > 0.05:
                s_tmp = []
                a_tmp = []
                r_tmp = []
                state = env.reset()
                done = False
        print("last reword",reward)
        r_tmp = Valuefunction.get_discounted_return(r_tmp)
        value_predict = []
        for index in range(len(s_tmp)):
            value_predict.append(Valuefunction.get_value(s_tmp[index]))
        advantage = r_tmp - value_predict
        Valuefunction.update(s_tmp,value_predict, r_tmp, lr*10)
        policy.update_policy(a_tmp, advantage, s_tmp, lr)
        record.append(np.sum(r_tmp))
        print("epoch",i,"total rewards",record[-1])
    return record

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    policy = DiscretePolicy(1, 1000)
    Valuefunction = ValueFunction(0.999)
    total_return = train(env, policy, Valuefunction, 500, 1e-3)

    plt.plot(total_return)
    plt.show()

    state = env.reset()
    done = False
    while done == False:
        a = policy.get_action(state)
        observation, reward, done, info = env.step(a)
        env.render()
    env.close()
    
