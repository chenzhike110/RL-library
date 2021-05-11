import gym
env = gym.make('Humanoid-v3')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
env = gym.wrappers.Monitor(env, './video/', force = True)
t = 0
while True:
    t += 1
    env.render()
    observation = env.reset()
    print(observation)

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        observation = env.reset()
    # break
env.close()