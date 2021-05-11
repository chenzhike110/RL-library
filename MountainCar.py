import gym
from gym import spaces
import time
env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.action_space.sample())
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# env.reset()

# for i in range(10000):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         print("over!")
#         break
#     env.render()
#     time.sleep(0.1)