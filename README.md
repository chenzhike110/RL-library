## Implementation of Reinforcement Learning Algorithm

This is my python library and notes for Reinforcement Learning. Hope I can understand these algorithms completely.

### Key Concepts

- Bellman Equations

  ![CodeCogsEqn](./image/bellman.svg)
  
  Q(s,a) is Action-value Function and V(s) is value Function
  
- Advantage Functions

  ![advantage](./image/advantage.svg)

- Policy Gradient

  ![policy_gradient](./image/policy_gradient.svg)

### Kinds of Algorithm

![rl_algorithms_9_15](./image/rl_algorithms_9_15.svg)

- Q-Learning

  ![Q-learning](./image/Q-learning.svg)
  
- Double Deep Q-Learning
  
  ![ddqn](./image/ddqn.svg)
  
  use different nets to choose action and estimate action-value function
  
- Dueling Deep Q-Learning

  ![duelingdqn](./image/duelingdqn.svg)

- A2C \ A3C

  ![A2C](/home/czk119/Desktop/RL-learning/image/A2C.svg)

- TD3 (Twin Delayed DDPG)

  ![td3_1](./image/td3_1.svg)

  ![td3_2](./image/td3_2.svg)

  ![td3_3](./image/td3_3.svg)

